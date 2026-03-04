/**
 * User Review: Dogfooding querymode as a new developer
 *
 * Goal: Exercise the full API surface, try edge cases, and document
 * every friction point, bug, and DX issue encountered.
 */

import * as path from "node:path";

// ============================================================================
// FRICTION #1: What do I import?
// The README says: import { QueryMode } from "querymode"
// But the "local" subpath is documented only in local.ts header comments.
// A new user would try the README import first and get Cloudflare DO errors.
// ============================================================================

// Using src/ directly since we're in the same repo (tsx handles .ts imports)
import { QueryMode, QueryModeError } from "../src/local.js";
import type { QueryResult, Row, FilterOp, ExplainResult } from "../src/local.js";

const PARQUET = path.resolve(import.meta.dirname ?? ".", "../wasm/benchmarks/benchmark_e2e.parquet");
const LANCE = path.resolve(import.meta.dirname ?? ".", "../wasm/benchmarks/benchmark_e2e.lance");

let issueCount = 0;
const issues: string[] = [];
function logIssue(category: string, description: string) {
  issueCount++;
  const entry = `${issueCount}. [${category}] ${description}`;
  issues.push(entry);
  console.log(`  >> ISSUE ${entry}`);
}

async function test(name: string, fn: () => Promise<void>) {
  console.log(`\n--- ${name} ---`);
  try {
    await fn();
    console.log(`  PASS`);
  } catch (err) {
    console.log(`  FAIL: ${err instanceof Error ? err.message : String(err)}`);
    if (err instanceof Error && err.stack) {
      // Print just the first 3 lines of stack
      const lines = err.stack.split("\n").slice(0, 4);
      for (const line of lines) console.log(`    ${line}`);
    }
  }
}

async function main() {
  console.log("=== QueryMode User Review ===");
  console.log(`Parquet file: ${PARQUET}`);
  console.log(`Lance file: ${LANCE}`);

  const qm = QueryMode.local();

  // =========================================================================
  // TEST 1: Basic query - does exec() return what I expect?
  // =========================================================================
  await test("1. Basic exec() shape", async () => {
    const result = await qm.table(PARQUET).limit(3).exec();
    console.log(`  rowCount=${result.rowCount}, columns=${result.columns}`);
    console.log(`  First row keys: ${Object.keys(result.rows[0] ?? {})}`);
    console.log(`  First row:`, result.rows[0]);

    // Check: are bigints returned for int columns? This is a DX concern.
    const firstRow = result.rows[0];
    if (firstRow) {
      for (const [key, val] of Object.entries(firstRow)) {
        if (typeof val === "bigint") {
          logIssue("DX", `Column "${key}" returns bigint (${val}n) instead of number. Users will need BigInt handling everywhere — JSON.stringify crashes on bigints.`);
          break; // Only report once
        }
      }
    }

    // Check: can I JSON.stringify the result?
    try {
      JSON.stringify(result);
    } catch (err) {
      logIssue("BUG", `JSON.stringify(result) fails: ${(err as Error).message}. QueryResult with bigints cannot be serialized, which breaks every HTTP API.`);
    }
  });

  // =========================================================================
  // TEST 2: Select non-existent column
  // =========================================================================
  await test("2. Select non-existent column", async () => {
    const result = await qm.table(PARQUET).select("nonexistent_column").exec();
    console.log(`  rowCount=${result.rowCount}, columns=${result.columns}`);
    console.log(`  First row:`, result.rows[0]);
    if (result.rowCount > 0 && Object.keys(result.rows[0] ?? {}).length === 0) {
      logIssue("DX", `Selecting a non-existent column silently returns empty row objects instead of throwing an error. Should fail fast with 'Column "nonexistent_column" not found in table'.`);
    }
    if (result.rowCount === 0) {
      logIssue("BUG", `Selecting a non-existent column returns 0 rows instead of throwing.`);
    }
  });

  // =========================================================================
  // TEST 3: Filter on non-existent column
  // =========================================================================
  await test("3. Filter on non-existent column", async () => {
    const result = await qm.table(PARQUET).filter("fake_col", "gt", 5).limit(5).exec();
    console.log(`  rowCount=${result.rowCount}`);
    if (result.rowCount > 0) {
      logIssue("DX", `Filter on non-existent column "fake_col" silently returns rows (filter ignored). Should throw or warn.`);
    }
  });

  // =========================================================================
  // TEST 4: Empty result set
  // =========================================================================
  await test("4. Empty result set", async () => {
    // Filter for amount > 999999 which shouldn't exist
    const result = await qm.table(PARQUET).filter("amount", "gt", 999999).exec();
    console.log(`  rowCount=${result.rowCount}, rows.length=${result.rows.length}`);
    if (result.rowCount !== result.rows.length) {
      logIssue("BUG", `rowCount (${result.rowCount}) !== rows.length (${result.rows.length})`);
    }
  });

  // =========================================================================
  // TEST 5: Type mismatch in filter value
  // =========================================================================
  await test("5. Type mismatch: filter string on numeric column", async () => {
    // "amount" is float64, filter with string value
    const result = await qm.table(PARQUET).filter("amount", "gt", "hello" as any).limit(5).exec();
    console.log(`  rowCount=${result.rowCount}`);
    logIssue("DX", `Filtering numeric column "amount" with string value "hello" does not throw — returns ${result.rowCount} rows. Should fail with type mismatch error.`);
  });

  // =========================================================================
  // TEST 6: .count() with no filter (metadata-only path)
  // =========================================================================
  await test("6. count() with no filter (metadata path)", async () => {
    const count = await qm.table(PARQUET).count();
    console.log(`  count=${count}`);
    if (count === 0) {
      logIssue("BUG", `count() with no filter returned 0`);
    }
  });

  // =========================================================================
  // TEST 7: .count() with filter
  // =========================================================================
  await test("7. count() with filter", async () => {
    const count = await qm.table(PARQUET).filter("amount", "gt", 500).count();
    console.log(`  count=${count}`);
  });

  // =========================================================================
  // TEST 8: .exists() true and false
  // =========================================================================
  await test("8. exists() true case", async () => {
    const e = await qm.table(PARQUET).filter("amount", "gt", 100).exists();
    console.log(`  exists=${e}`);
    if (!e) logIssue("BUG", `exists() returned false for amount > 100`);
  });

  await test("8b. exists() false case", async () => {
    const e = await qm.table(PARQUET).filter("amount", "gt", 999999).exists();
    console.log(`  exists=${e}`);
    if (e) logIssue("BUG", `exists() returned true for amount > 999999`);
  });

  // =========================================================================
  // TEST 9: .first()
  // =========================================================================
  await test("9. first() with match", async () => {
    const row = await qm.table(PARQUET).filter("amount", "gt", 999).first();
    console.log(`  first row:`, row);
    if (!row) logIssue("BUG", `first() returned null for amount > 999`);
  });

  await test("9b. first() with no match", async () => {
    const row = await qm.table(PARQUET).filter("amount", "gt", 999999).first();
    console.log(`  first row: ${row}`);
    if (row !== null) logIssue("BUG", `first() should return null for no match, got: ${JSON.stringify(row)}`);
  });

  // =========================================================================
  // TEST 10: .explain()
  // =========================================================================
  await test("10. explain() returns plan", async () => {
    const plan = await qm.table(PARQUET).filter("amount", "gt", 500).select("id", "amount").explain();
    console.log(`  format=${plan.format}, totalRows=${plan.totalRows}, estimatedRows=${plan.estimatedRows}`);
    console.log(`  pages: ${plan.pagesScanned}/${plan.pagesTotal} scanned`);
    console.log(`  filters:`, plan.filters);

    // Check: explain says 0 pages skipped even though filter should enable pushdown
    if (plan.pagesSkipped === 0 && plan.pagesTotal > 1) {
      logIssue("DX", `explain() shows 0 pages skipped despite filter on "amount". Is page-level min/max pushdown working for Parquet?`);
    }
  });

  // =========================================================================
  // TEST 11: .offset() pagination
  // =========================================================================
  await test("11. offset() pagination", async () => {
    const page1 = await qm.table(PARQUET).select("id", "amount").sort("id", "asc").limit(5).exec();
    const page2 = await qm.table(PARQUET).select("id", "amount").sort("id", "asc").limit(5).offset(5).exec();
    console.log(`  Page 1 ids: ${page1.rows.map(r => r.id)}`);
    console.log(`  Page 2 ids: ${page2.rows.map(r => r.id)}`);

    // Check for overlap
    const p1ids = new Set(page1.rows.map(r => String(r.id)));
    const overlap = page2.rows.filter(r => p1ids.has(String(r.id)));
    if (overlap.length > 0) {
      logIssue("BUG", `offset() pagination has overlapping rows between page 1 and page 2`);
    }
  });

  // =========================================================================
  // TEST 12: .after() keyset pagination
  // =========================================================================
  await test("12. after() keyset pagination", async () => {
    const page1 = await qm.table(PARQUET).select("id", "amount").sort("id", "asc").limit(5).exec();
    const lastId = page1.rows[page1.rows.length - 1]?.id;
    console.log(`  Page 1 last id: ${lastId}`);

    if (lastId != null) {
      const page2 = await qm.table(PARQUET).select("id", "amount").sort("id", "asc").after(lastId).limit(5).exec();
      console.log(`  Page 2 first id: ${page2.rows[0]?.id}`);

      // Verify no overlap and correct ordering
      if (page2.rows[0] && page2.rows[0].id !== undefined) {
        const firstP2 = Number(page2.rows[0].id);
        const lastP1 = Number(lastId);
        if (firstP2 <= lastP1) {
          logIssue("BUG", `after() keyset pagination: page 2 first id (${firstP2}) <= page 1 last id (${lastP1})`);
        }
      }
    }
  });

  // =========================================================================
  // TEST 13: .after() without .sort() should throw
  // =========================================================================
  await test("13. after() without sort() throws", async () => {
    try {
      qm.table(PARQUET).after(5);
      logIssue("BUG", `after() without sort() did not throw`);
    } catch (err) {
      console.log(`  Correctly threw: ${(err as Error).message}`);
    }
  });

  // =========================================================================
  // TEST 14: .cache() works in local mode
  // =========================================================================
  await test("14. cache() second call is faster", async () => {
    const q = () => qm.table(PARQUET).filter("amount", "gt", 800).select("id").limit(10).cache({ ttl: 5000 });

    const t1 = Date.now();
    const r1 = await q().exec();
    const d1 = Date.now() - t1;

    const t2 = Date.now();
    const r2 = await q().exec();
    const d2 = Date.now() - t2;

    console.log(`  First: ${d1}ms (cacheHit=${r1.cacheHit}), Second: ${d2}ms (cacheHit=${r2.cacheHit})`);
    if (!r2.cacheHit) {
      logIssue("BUG", `cache() second call did not hit cache`);
    }
  });

  // =========================================================================
  // TEST 15: Chaining in unusual order
  // =========================================================================
  await test("15. Chaining: limit before filter", async () => {
    // Does order matter? limit().filter() vs filter().limit()
    const r1 = await qm.table(PARQUET).filter("amount", "gt", 500).limit(5).exec();
    const r2 = await qm.table(PARQUET).limit(5).filter("amount", "gt", 500).exec();
    console.log(`  filter-then-limit: ${r1.rowCount} rows`);
    console.log(`  limit-then-filter: ${r2.rowCount} rows`);
    if (r1.rowCount !== r2.rowCount) {
      logIssue("DX", `Chaining order matters: filter().limit() gives ${r1.rowCount} rows, limit().filter() gives ${r2.rowCount} rows. This is surprising — most builders are order-independent.`);
    }
  });

  // =========================================================================
  // TEST 16: Multiple filters (AND semantics?)
  // =========================================================================
  await test("16. Multiple filters (AND semantics)", async () => {
    const r = await qm.table(PARQUET)
      .filter("amount", "gt", 500)
      .filter("amount", "lt", 600)
      .select("id", "amount")
      .limit(10)
      .exec();
    console.log(`  Rows with 500 < amount < 600: ${r.rowCount}`);
    if (r.rowCount > 0) {
      const allInRange = r.rows.every(row => {
        const amt = Number(row.amount);
        return amt > 500 && amt < 600;
      });
      if (!allInRange) {
        logIssue("BUG", `Multiple filters not applied as AND — some rows outside range`);
      }
    }
  });

  // =========================================================================
  // TEST 17: .where() alias for .filter()
  // =========================================================================
  await test("17. where() is alias for filter()", async () => {
    const r1 = await qm.table(PARQUET).filter("amount", "gt", 900).limit(5).exec();
    const r2 = await qm.table(PARQUET).where("amount", "gt", 900).limit(5).exec();
    console.log(`  filter(): ${r1.rowCount} rows, where(): ${r2.rowCount} rows`);
    // They should be identical
    if (r1.rowCount !== r2.rowCount) {
      logIssue("BUG", `where() and filter() return different results`);
    }
  });

  // =========================================================================
  // TEST 18: "in" operator
  // =========================================================================
  await test("18. 'in' operator", async () => {
    // Get some known IDs first
    const sample = await qm.table(PARQUET).select("id").limit(5).exec();
    const ids = sample.rows.map(r => Number(r.id));
    console.log(`  Looking for ids: ${ids}`);

    const r = await qm.table(PARQUET).filter("id", "in", ids).exec();
    console.log(`  Found: ${r.rowCount} rows`);
    if (r.rowCount !== ids.length) {
      logIssue("DX", `"in" filter for ${ids.length} known ids returned ${r.rowCount} rows (expected ${ids.length}). Possible bigint/number comparison issue.`);
    }
  });

  // =========================================================================
  // TEST 19: "neq" operator
  // =========================================================================
  await test("19. 'neq' operator", async () => {
    // Get first row's id, then filter neq
    const first = await qm.table(PARQUET).select("id").limit(1).exec();
    const firstId = first.rows[0]?.id;
    console.log(`  Filtering neq id=${firstId}`);

    const count = await qm.table(PARQUET).filter("id", "neq", Number(firstId)).count();
    const totalCount = await qm.table(PARQUET).count();
    console.log(`  Total: ${totalCount}, After neq: ${count}`);
    if (count !== totalCount - 1) {
      logIssue("DX", `neq filter: expected ${totalCount - 1}, got ${count}. Off by ${totalCount - 1 - count}.`);
    }
  });

  // =========================================================================
  // TEST 20: Nonexistent file — error quality
  // =========================================================================
  await test("20. Nonexistent file error", async () => {
    try {
      await qm.table("/tmp/does-not-exist.parquet").exec();
      logIssue("BUG", `No error thrown for nonexistent file`);
    } catch (err) {
      if (err instanceof QueryModeError) {
        console.log(`  code=${err.code}, message=${err.message}`);
        if (err.code !== "TABLE_NOT_FOUND") {
          logIssue("DX", `Expected error code TABLE_NOT_FOUND, got ${err.code}`);
        }
      } else {
        logIssue("DX", `Error is not a QueryModeError, it's ${(err as Error).constructor.name}: ${(err as Error).message}`);
      }
    }
  });

  // =========================================================================
  // TEST 21: Invalid file (not parquet/lance)
  // =========================================================================
  await test("21. Invalid file format", async () => {
    try {
      // Pass this script file as a "table"
      await qm.table(path.resolve(import.meta.dirname ?? ".", "user-review.ts")).exec();
      logIssue("BUG", `No error thrown for non-parquet/lance file`);
    } catch (err) {
      if (err instanceof QueryModeError) {
        console.log(`  code=${err.code}, message=${err.message}`);
      } else {
        console.log(`  Error type: ${(err as Error).constructor.name}, message: ${(err as Error).message}`);
        logIssue("DX", `Invalid format error is not a QueryModeError — got ${(err as Error).constructor.name}`);
      }
    }
  });

  // =========================================================================
  // TEST 22: Aggregation without groupBy
  // =========================================================================
  await test("22. Aggregation (sum, avg, min, max, count) without groupBy", async () => {
    const r = await qm.table(PARQUET)
      .aggregate("sum", "amount", "total")
      .aggregate("avg", "amount", "average")
      .aggregate("min", "amount", "minimum")
      .aggregate("max", "amount", "maximum")
      .aggregate("count", "*", "total_count")
      .exec();
    console.log(`  Result:`, r.rows[0]);
    const row = r.rows[0];
    if (row) {
      if (row.total === undefined) logIssue("BUG", `sum aggregate missing from result`);
      if (row.average === undefined) logIssue("BUG", `avg aggregate missing from result`);
      if (row.minimum === undefined) logIssue("BUG", `min aggregate missing from result`);
      if (row.maximum === undefined) logIssue("BUG", `max aggregate missing from result`);
      if (row.total_count === undefined) logIssue("BUG", `count aggregate missing from result`);
    }
  });

  // =========================================================================
  // TEST 23: groupBy + aggregate
  // =========================================================================
  await test("23. groupBy + aggregate", async () => {
    const r = await qm.table(PARQUET)
      .groupBy("customer_id")
      .aggregate("count", "*", "n")
      .aggregate("sum", "amount", "total")
      .limit(5)
      .exec();
    console.log(`  Groups: ${r.rowCount}`);
    console.log(`  Sample:`, r.rows.slice(0, 3));
  });

  // =========================================================================
  // TEST 24: sort() without limit() (full sort)
  // =========================================================================
  await test("24. sort() without limit() on small filter set", async () => {
    const r = await qm.table(PARQUET)
      .filter("amount", "gt", 998)
      .select("id", "amount")
      .sort("amount", "desc")
      .exec();
    console.log(`  Rows: ${r.rowCount}`);
    if (r.rowCount > 1) {
      const sorted = r.rows.every((row, i) => i === 0 || Number(row.amount) <= Number(r.rows[i - 1].amount));
      if (!sorted) logIssue("BUG", `sort("amount", "desc") did not sort correctly`);
      else console.log(`  Sort order verified correct`);
    }
  });

  // =========================================================================
  // TEST 25: sort("asc") default
  // =========================================================================
  await test("25. sort() default direction is asc", async () => {
    const r = await qm.table(PARQUET)
      .filter("amount", "gt", 998)
      .select("id", "amount")
      .sort("amount")
      .exec();
    if (r.rowCount > 1) {
      const sorted = r.rows.every((row, i) => i === 0 || Number(row.amount) >= Number(r.rows[i - 1].amount));
      if (!sorted) logIssue("BUG", `sort("amount") without direction did not sort ascending`);
      else console.log(`  Default asc sort verified correct`);
    }
  });

  // =========================================================================
  // TEST 26: Cursor / streaming
  // =========================================================================
  await test("26. cursor() iteration", async () => {
    let totalRows = 0;
    let batches = 0;
    for await (const batch of qm.table(PARQUET).filter("amount", "gt", 900).select("id", "amount").cursor({ batchSize: 1000 })) {
      totalRows += batch.length;
      batches++;
      if (batches >= 3) break; // Don't iterate everything
    }
    console.log(`  Iterated ${batches} batches, ${totalRows} total rows`);
  });

  // =========================================================================
  // TEST 27: Lance file query
  // =========================================================================
  await test("27. Query Lance file", async () => {
    try {
      const r = await qm.table(LANCE).limit(5).exec();
      console.log(`  Lance rowCount=${r.rowCount}, columns=${r.columns}`);
      if (r.rowCount === 0 && r.columns.length > 0) {
        logIssue("BUG", `Lance file has columns (${r.columns.join(", ")}) but returns 0 rows. Data not being decoded?`);
      }
    } catch (err) {
      console.log(`  Lance query error: ${(err as Error).message}`);
      // This might be expected if LANCE is a directory
    }
  });

  // =========================================================================
  // TEST 28: TypeScript type inference
  // =========================================================================
  await test("28. TypeScript type checking (compile-time only)", async () => {
    // These are compile-time checks — if this file compiles, types work
    const result: QueryResult = await qm.table(PARQUET).limit(1).exec();
    const row: Row | null = await qm.table(PARQUET).first();
    const count: number = await qm.table(PARQUET).count();
    const exists: boolean = await qm.table(PARQUET).exists();
    const plan: ExplainResult = await qm.table(PARQUET).explain();

    // Check that Row type is Record<string, ...> — is it useful?
    if (row) {
      // TypeScript says row.id is `number | bigint | string | boolean | Float32Array | null`
      // which is correct but not very helpful — user needs runtime checks
      const id = row.id; // type is too wide to be useful
      logIssue("DX", `Row type is Record<string, number|bigint|string|boolean|Float32Array|null>. No per-column types. Users must cast everything.`);
    }
    console.log(`  Types compile correctly`);
  });

  // =========================================================================
  // TEST 29: select() with empty array
  // =========================================================================
  await test("29. select() with no args", async () => {
    const r = await qm.table(PARQUET).select().limit(3).exec();
    console.log(`  columns=${r.columns}, row keys=${Object.keys(r.rows[0] ?? {})}`);
    // select() with no args should return all columns
    if (r.columns.length === 0) {
      logIssue("DX", `select() with no arguments returns 0 columns. Should either return all columns or throw.`);
    }
  });

  // =========================================================================
  // TEST 30: Double exec()
  // =========================================================================
  await test("30. Double exec() on same builder", async () => {
    const q = qm.table(PARQUET).limit(3);
    const r1 = await q.exec();
    const r2 = await q.exec();
    console.log(`  First exec: ${r1.rowCount} rows, Second exec: ${r2.rowCount} rows`);
    // Should be safe to call twice
    if (r1.rowCount !== r2.rowCount) {
      logIssue("BUG", `Double exec() returns different results`);
    }
  });

  // =========================================================================
  // TEST 31: execStream() in local mode
  // =========================================================================
  await test("31. execStream() in local mode", async () => {
    try {
      const stream = await qm.table(PARQUET).limit(5).execStream();
      logIssue("DX", `execStream() did not throw in local mode — should it work?`);
    } catch (err) {
      console.log(`  Correctly threw: ${(err as Error).message}`);
    }
  });

  // =========================================================================
  // TEST 32: append() in local mode (on parquet file)
  // =========================================================================
  await test("32. append() on parquet file", async () => {
    try {
      await qm.table(PARQUET).append([{ id: 1, name: "test" }]);
      logIssue("DX", `append() did not throw on a parquet file — should it only work on Lance dirs?`);
    } catch (err) {
      console.log(`  Error: ${(err as Error).message}`);
    }
  });

  // =========================================================================
  // TEST 33: Performance — is metadata cached across queries?
  // =========================================================================
  await test("33. Metadata caching (second query faster)", async () => {
    // First query loads metadata
    const t1 = Date.now();
    await qm.table(PARQUET).limit(1).exec();
    const d1 = Date.now() - t1;

    // Second query should reuse cached metadata
    const t2 = Date.now();
    await qm.table(PARQUET).limit(1).exec();
    const d2 = Date.now() - t2;

    console.log(`  First: ${d1}ms, Second: ${d2}ms`);
  });

  // =========================================================================
  // TEST 34: "eq" filter with bigint value (type mismatch)
  // =========================================================================
  await test("34. Filter with bigint value", async () => {
    // The id column returns bigint. Can I filter with bigint?
    const r = await qm.table(PARQUET).filter("id", "eq", 1n as any).limit(5).exec();
    console.log(`  rowCount=${r.rowCount}`);
    if (r.rowCount === 0) {
      // Try with number
      const r2 = await qm.table(PARQUET).filter("id", "eq", 1).limit(5).exec();
      console.log(`  With number filter: ${r2.rowCount}`);
      if (r2.rowCount > 0) {
        logIssue("DX", `Filter with bigint value returns 0 rows, but number value works. The column returns bigints but you must filter with numbers. This is confusing.`);
      }
    }
  });

  // =========================================================================
  // TEST 35: .query() orchestration method
  // =========================================================================
  await test("35. .query() orchestration", async () => {
    // The QueryMode.query() method just runs a function — is it useful?
    const result = await qm.query(async () => {
      const orders = await qm.table(PARQUET).filter("amount", "gt", 900).select("id", "amount").limit(5).exec();
      return orders;
    });
    console.log(`  query() returned ${result.rowCount} rows`);
    logIssue("DX", `.query() just calls the function — it provides no transaction isolation, no shared context, no optimization. Consider removing or documenting why it exists.`);
  });

  // =========================================================================
  // TEST 36: Negative limit
  // =========================================================================
  await test("36. Negative limit", async () => {
    try {
      const r = await qm.table(PARQUET).limit(-1).exec();
      console.log(`  Returned ${r.rowCount} rows with limit(-1)`);
      logIssue("DX", `limit(-1) does not throw — returns ${r.rowCount} rows. Should validate positive integer.`);
    } catch (err) {
      console.log(`  Correctly threw: ${(err as Error).message}`);
    }
  });

  // =========================================================================
  // TEST 37: limit(0)
  // =========================================================================
  await test("37. limit(0)", async () => {
    try {
      const r = await qm.table(PARQUET).limit(0).exec();
      console.log(`  Returned ${r.rowCount} rows with limit(0)`);
      if (r.rowCount > 0) {
        logIssue("DX", `limit(0) returns ${r.rowCount} rows. Should return 0 rows or throw.`);
      }
    } catch (err) {
      console.log(`  Threw: ${(err as Error).message}`);
    }
  });

  // =========================================================================
  // TEST 38: offset without limit
  // =========================================================================
  await test("38. offset() without limit()", async () => {
    const r = await qm.table(PARQUET).select("id").offset(99990).exec();
    console.log(`  Rows after offset 99990: ${r.rowCount}`);
    if (r.rowCount > 100) {
      logIssue("BUG", `offset(99990) without limit returns ${r.rowCount} rows (expected ~10). Offset is not being applied.`);
    }
  });

  // =========================================================================
  // TEST 39: append() error message quality
  // =========================================================================
  await test("39. append() error message on parquet file", async () => {
    try {
      await qm.table(PARQUET).append([{ id: 1, name: "test" }]);
    } catch (err) {
      const msg = (err as Error).message;
      if (msg.includes("ENOTDIR")) {
        logIssue("DX", `append() on a parquet file gives low-level ENOTDIR error. Should say "append() only works on Lance dataset directories, not single files".`);
      }
    }
  });

  // =========================================================================
  // TEST 40: filter on non-existent column
  // =========================================================================
  await test("40. Filter on non-existent column returns 0 not error", async () => {
    const totalCount = await qm.table(PARQUET).count();
    const r = await qm.table(PARQUET).filter("totally_fake", "eq", 1).count();
    console.log(`  Total rows: ${totalCount}, After fake filter: ${r}`);
    if (r === 0) {
      logIssue("DX", `filter("totally_fake", "eq", 1) silently returns 0 rows instead of throwing "Column not found". The user gets no feedback that their column name is wrong.`);
    }
  });

  // =========================================================================
  // TEST 41: groupBy with sort on aggregate alias
  // =========================================================================
  await test("41. groupBy + sort on aggregate alias", async () => {
    const r = await qm.table(PARQUET)
      .groupBy("customer_id")
      .aggregate("sum", "amount", "total_revenue")
      .sort("total_revenue", "desc")
      .limit(5)
      .exec();
    console.log(`  Rows: ${r.rowCount}`);
    if (r.rowCount > 1) {
      const sorted = r.rows.every((row, i) => i === 0 || Number(row.total_revenue) <= Number(r.rows[i - 1].total_revenue));
      if (!sorted) {
        console.log(`  Sort order:`, r.rows.map(row => row.total_revenue));
        logIssue("BUG", `groupBy + sort on aggregate alias "total_revenue" does not sort correctly`);
      } else {
        console.log(`  Sort order verified correct`);
      }
    }
  });

  // =========================================================================
  // TEST 42: README import path test
  // =========================================================================
  await test("42. README import discoverability", async () => {
    // The README shows: import { QueryMode } from "querymode"
    // But that imports the Cloudflare DO version which crashes in Node.
    // The local import path is not mentioned in the README Query API section.
    logIssue("DX", `README "Query API" section shows 'import { QueryMode } from "querymode"' but local usage requires 'import { QueryMode } from "querymode/local"'. The local import path is not documented in the README.`);
  });

  // =========================================================================
  // TEST 43: Verify cacheHit on first call is undefined vs false
  // =========================================================================
  await test("43. cache() first call cacheHit field", async () => {
    // Use unique filter to avoid hitting existing cache
    const r = await qm.table(PARQUET).filter("amount", "gt", 777.777).select("id").limit(1).cache({ ttl: 1000 }).exec();
    console.log(`  cacheHit=${r.cacheHit} (type=${typeof r.cacheHit})`);
    if (r.cacheHit === undefined) {
      logIssue("DX", `First cached query returns cacheHit=undefined instead of false. Makes boolean checks unreliable (if (result.cacheHit) vs if (result.cacheHit === true)).`);
    }
  });

  // =========================================================================
  // SUMMARY
  // =========================================================================
  console.log("\n" + "=".repeat(70));
  console.log("USER REVIEW SUMMARY");
  console.log("=".repeat(70));
  console.log(`\nTotal issues found: ${issueCount}\n`);
  for (const issue of issues) {
    console.log(issue);
  }
  console.log();
}

main().catch(err => {
  console.error("FATAL:", err);
  process.exit(1);
});
