import { describe, it, expect } from "vitest";
import { kWayMerge, mergeQueryResults } from "./merge.js";
import type { QueryResult } from "./types.js";
import type { QueryDescriptor } from "./client.js";

describe("kWayMerge", () => {
  it("merges ascending with limit", () => {
    const arrays = [
      [{ v: 1 }, { v: 3 }, { v: 5 }],
      [{ v: 2 }, { v: 4 }, { v: 6 }],
      [{ v: 0 }, { v: 7 }],
    ];
    const result = kWayMerge(arrays, "v", "asc", 5);
    expect(result.map((r) => r.v)).toEqual([0, 1, 2, 3, 4]);
  });

  it("merges descending with limit", () => {
    const arrays = [
      [{ v: 5 }, { v: 3 }, { v: 1 }],
      [{ v: 6 }, { v: 4 }, { v: 2 }],
      [{ v: 7 }, { v: 0 }],
    ];
    const result = kWayMerge(arrays, "v", "desc", 3);
    expect(result.map((r) => r.v)).toEqual([7, 6, 5]);
  });

  it("handles empty arrays", () => {
    const arrays = [[], [{ v: 1 }], [], [{ v: 2 }]];
    const result = kWayMerge(arrays, "v", "asc", 10);
    expect(result.map((r) => r.v)).toEqual([1, 2]);
  });
});

function makeResult(rows: Record<string, number>[]): QueryResult {
  return {
    rows,
    rowCount: rows.length,
    columns: rows.length > 0 ? Object.keys(rows[0]) : [],
    bytesRead: 100,
    pagesSkipped: 2,
    durationMs: 10,
  };
}

describe("mergeQueryResults", () => {
  it("concatenates rows when no sort", () => {
    const p1 = makeResult([{ v: 1 }, { v: 2 }]);
    const p2 = makeResult([{ v: 3 }]);
    const query: QueryDescriptor = {
      table: "t",
      filters: [],
      projections: ["v"],
    };

    const merged = mergeQueryResults([p1, p2], query);
    expect(merged.rows).toHaveLength(3);
    expect(merged.rows.map((r) => r.v)).toEqual([1, 2, 3]);
  });

  it("uses kWayMerge when sort + limit", () => {
    const p1 = makeResult([{ v: 1 }, { v: 3 }]);
    const p2 = makeResult([{ v: 2 }, { v: 4 }]);
    const query: QueryDescriptor = {
      table: "t",
      filters: [],
      projections: ["v"],
      sortColumn: "v",
      sortDirection: "asc",
      limit: 3,
    };

    const merged = mergeQueryResults([p1, p2], query);
    expect(merged.rows.map((r) => r.v)).toEqual([1, 2, 3]);
  });

  it("sums bytesRead across partials", () => {
    const p1 = makeResult([{ v: 1 }]);
    const p2 = makeResult([{ v: 2 }]);
    const query: QueryDescriptor = {
      table: "t",
      filters: [],
      projections: ["v"],
    };

    const merged = mergeQueryResults([p1, p2], query);
    expect(merged.bytesRead).toBe(200);
    expect(merged.pagesSkipped).toBe(4);
  });

  it("handles large number of partials (simulates hierarchical reduction input)", () => {
    // Simulate 100 fragment results — this is what a reducer DO would merge
    const partials = Array.from({ length: 100 }, (_, i) =>
      makeResult([{ v: i * 2 }, { v: i * 2 + 1 }]),
    );
    const query: QueryDescriptor = {
      table: "t",
      filters: [],
      projections: ["v"],
      sortColumn: "v",
      sortDirection: "asc",
      limit: 10,
    };

    const merged = mergeQueryResults(partials, query);
    expect(merged.rows.map(r => r.v)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    expect(merged.bytesRead).toBe(10000); // 100 × 100
  });

  it("aggregation merge works across many partials", () => {
    const partials = Array.from({ length: 50 }, (_, i) =>
      makeResult([{ group: "a", val: i + 1 }]),
    );
    const query: QueryDescriptor = {
      table: "t",
      filters: [],
      projections: [],
      aggregates: [{ fn: "sum", column: "val" }],
      groupBy: ["group"],
    };

    const merged = mergeQueryResults(partials, query);
    expect(merged.rows).toHaveLength(1);
    // Sum of 1..50 = 1275
    expect(merged.rows[0].sum_val).toBe(1275);
  });

  it("propagates telemetry fields from partials", () => {
    const p1: QueryResult = {
      rows: [{ v: 1 }], rowCount: 1, columns: ["v"],
      bytesRead: 100, pagesSkipped: 2, durationMs: 10,
      r2ReadMs: 5, wasmExecMs: 3, cacheHits: 4, cacheMisses: 1,
      edgeCacheHits: 2, edgeCacheMisses: 0, spillBytesWritten: 1024, spillBytesRead: 512,
    };
    const p2: QueryResult = {
      rows: [{ v: 2 }], rowCount: 1, columns: ["v"],
      bytesRead: 200, pagesSkipped: 3, durationMs: 15,
      r2ReadMs: 8, wasmExecMs: 6, cacheHits: 3, cacheMisses: 2,
      edgeCacheHits: 1, edgeCacheMisses: 1, spillBytesWritten: 2048, spillBytesRead: 1024,
    };
    const query: QueryDescriptor = { table: "t", filters: [], projections: ["v"] };
    const merged = mergeQueryResults([p1, p2], query);
    expect(merged.bytesRead).toBe(300);
    expect(merged.pagesSkipped).toBe(5);
    expect(merged.durationMs).toBe(15);
    // Timing: max across partials (parallel execution)
    expect(merged.r2ReadMs).toBe(8);
    expect(merged.wasmExecMs).toBe(6);
    // Counters: summed
    expect(merged.cacheHits).toBe(7);
    expect(merged.cacheMisses).toBe(3);
    expect(merged.edgeCacheHits).toBe(3);
    expect(merged.edgeCacheMisses).toBe(1);
    expect(merged.spillBytesWritten).toBe(3072);
    expect(merged.spillBytesRead).toBe(1536);
  });

  it("omits telemetry fields when no partials have them", () => {
    const p1 = makeResult([{ v: 1 }]);
    const query: QueryDescriptor = { table: "t", filters: [], projections: ["v"] };
    const merged = mergeQueryResults([p1], query);
    expect(merged.r2ReadMs).toBeUndefined();
    expect(merged.wasmExecMs).toBeUndefined();
    expect(merged.cacheHits).toBeUndefined();
    expect(merged.spillBytesWritten).toBeUndefined();
  });

  it("offset works with sorted merge across partials", () => {
    const partials = Array.from({ length: 10 }, (_, i) =>
      makeResult([{ v: i }]),
    );
    const query: QueryDescriptor = {
      table: "t",
      filters: [],
      projections: ["v"],
      sortColumn: "v",
      sortDirection: "asc",
      limit: 3,
      offset: 5,
    };

    const merged = mergeQueryResults(partials, query);
    expect(merged.rows.map(r => r.v)).toEqual([5, 6, 7]);
  });
});
