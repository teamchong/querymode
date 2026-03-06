/**
 * QueryMode local quickstart — run with: npx tsx examples/local-quickstart.ts
 *
 * Reads a Lance or Parquet file from disk, applies filter + sort + limit,
 * and prints the result. Replace the path with your own data file.
 */
import { LocalExecutor } from "../src/local-executor.js";
import { DataFrame } from "../src/client.js";

const executor = new LocalExecutor();
const TABLE = process.argv[2] ?? "./data/events.parquet";

const df = new DataFrame(TABLE, executor);

const result = await df
  .filter("amount", "gt", 100)
  .whereNotNull("region")
  .select("id", "amount", "region")
  .sort("amount", "desc")
  .limit(10)
  .collect();

console.log(`${result.rowCount} rows, ${result.pagesSkipped} pages skipped`);
console.table(result.rows);
