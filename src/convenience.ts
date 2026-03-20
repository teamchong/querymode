/**
 * Convenience factories for zero-config QueryMode usage.
 * No files needed — create DataFrames from JSON arrays, CSV strings, or demo data.
 */

import { DataFrame, MaterializedExecutor } from "./client.js";
import type { QueryResult, Row } from "./types.js";

/**
 * Create a DataFrame from an array of plain objects.
 * Coerces values to Row-compatible types.
 */
export function createFromJSON<T extends Record<string, unknown>>(
  data: T[],
  tableName = "__json__",
): DataFrame {
  if (data.length === 0) {
    const result: QueryResult = { rows: [], rowCount: 0, columns: [], bytesRead: 0, pagesSkipped: 0, durationMs: 0 };
    return new DataFrame(tableName, new MaterializedExecutor(result));
  }

  const columnSet = new Set<string>();
  for (const item of data) {
    for (const key of Object.keys(item)) columnSet.add(key);
  }
  const columns = [...columnSet];
  const rows: Row[] = data.map(item => {
    const row: Row = {};
    for (const col of columns) {
      const v = item[col];
      if (v === null || v === undefined) {
        row[col] = null;
      } else if (typeof v === "number" || typeof v === "bigint" || typeof v === "string" || typeof v === "boolean") {
        row[col] = v;
      } else if (v instanceof Float32Array) {
        row[col] = v;
      } else {
        row[col] = JSON.stringify(v);
      }
    }
    return row;
  });

  const result: QueryResult = {
    rows,
    rowCount: rows.length,
    columns,
    bytesRead: 0,
    pagesSkipped: 0,
    durationMs: 0,
  };
  return new DataFrame(tableName, new MaterializedExecutor(result));
}

/**
 * Create a DataFrame from a CSV string.
 * Auto-detects delimiter (comma, tab, pipe, semicolon) and infers column types.
 */
export async function createFromCSV(csv: string, tableName = "__csv__"): Promise<DataFrame> {
  const { parseCsvFull } = await import("./readers/csv-reader.js");
  const parsed = parseCsvFull(csv);

  if (parsed.rowCount === 0) {
    const result: QueryResult = { rows: [], rowCount: 0, columns: parsed.headers, bytesRead: 0, pagesSkipped: 0, durationMs: 0 };
    return new DataFrame(tableName, new MaterializedExecutor(result));
  }

  const rows: Row[] = [];
  for (let r = 0; r < parsed.rowCount; r++) {
    const row: Row = {};
    for (let c = 0; c < parsed.headers.length; c++) {
      row[parsed.headers[c]] = parsed.columns[c][r] as Row[string];
    }
    rows.push(row);
  }

  const result: QueryResult = {
    rows,
    rowCount: rows.length,
    columns: parsed.headers,
    bytesRead: 0,
    pagesSkipped: 0,
    durationMs: 0,
  };
  return new DataFrame(tableName, new MaterializedExecutor(result));
}

/**
 * Create a demo DataFrame with 1000 rows of deterministic sample data.
 * No files needed — great for trying out the API.
 *
 * Columns: id (number), region (string), category (string), amount (number), created_at (string)
 */
export function createDemo(tableName = "__demo__"): DataFrame {
  const regions = ["us-east", "us-west", "eu-west", "eu-central", "ap-south", "ap-east"];
  const categories = ["Electronics", "Books", "Clothing", "Home", "Sports"];

  // Deterministic xorshift32 PRNG (same results every call)
  let state = 42;
  const xorshift32 = (): number => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0);
  };

  const rows: Row[] = [];
  for (let i = 0; i < 1000; i++) {
    const r1 = xorshift32();
    const region = regions[r1 % regions.length];
    const r2 = xorshift32();
    const category = categories[r2 % categories.length];
    const r3 = xorshift32();
    const amount = (r3 % 10000) / 100;
    const r4 = xorshift32();
    const day = (r4 % 365) + 1;
    const date = new Date(2024, 0, day);
    const created_at = date.toISOString().split("T")[0];

    rows.push({ id: i + 1, region, category, amount, created_at });
  }

  const result: QueryResult = {
    rows,
    rowCount: rows.length,
    columns: ["id", "region", "category", "amount", "created_at"],
    bytesRead: 0,
    pagesSkipped: 0,
    durationMs: 0,
  };
  return new DataFrame(tableName, new MaterializedExecutor(result));
}
