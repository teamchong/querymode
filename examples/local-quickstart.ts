/**
 * QueryMode local quickstart — run with: npx tsx examples/local-quickstart.ts
 *
 * Reads a Lance or Parquet file from disk, applies filter + sort + limit,
 * and prints the result. Replace the path with your own data file.
 */
import { QueryMode } from "../src/local.js";

const TABLE = process.argv[2] ?? "./data/events.parquet";

const qm = QueryMode.local();
const result = await qm
  .table(TABLE)
  .filter("amount", "gt", 100)
  .whereNotNull("region")
  .select("id", "amount", "region")
  .sort("amount", "desc")
  .limit(10)
  .collect();

console.log(`${result.rowCount} rows, ${result.pagesSkipped} pages skipped`);
console.table(result.rows);
