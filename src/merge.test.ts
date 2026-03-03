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
});
