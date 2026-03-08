import type { Row, QueryResult } from "./types.js";
import type { QueryDescriptor } from "./client.js";
import {
  computePartialAgg,
  mergePartialAggs,
  finalizePartialAgg,
} from "./partial-agg.js";

interface HeapEntry {
  arrayIdx: number;
  row: Row;
}

function siftDown(
  heap: HeapEntry[],
  i: number,
  sortCol: string,
  asc: boolean,
): void {
  const size = heap.length;
  while (true) {
    let target = i;
    const left = 2 * i + 1;
    const right = 2 * i + 2;
    if (left < size && compare(heap[left], heap[target], sortCol, asc) < 0) {
      target = left;
    }
    if (right < size && compare(heap[right], heap[target], sortCol, asc) < 0) {
      target = right;
    }
    if (target === i) break;
    [heap[i], heap[target]] = [heap[target], heap[i]];
    i = target;
  }
}

function siftUp(
  heap: HeapEntry[],
  i: number,
  sortCol: string,
  asc: boolean,
): void {
  while (i > 0) {
    const parent = (i - 1) >> 1;
    if (compare(heap[i], heap[parent], sortCol, asc) < 0) {
      [heap[i], heap[parent]] = [heap[parent], heap[i]];
      i = parent;
    } else {
      break;
    }
  }
}

function compare(
  a: HeapEntry,
  b: HeapEntry,
  sortCol: string,
  asc: boolean,
): number {
  const va = a.row[sortCol];
  const vb = b.row[sortCol];
  if (va === vb) return 0;
  if (va === null || va === undefined) return 1;
  if (vb === null || vb === undefined) return -1;
  const cmp = va < vb ? -1 : 1;
  return asc ? cmp : -cmp;
}

export function kWayMerge(
  arrays: Row[][],
  sortCol: string,
  dir: "asc" | "desc",
  limit: number,
): Row[] {
  const asc = dir === "asc";
  const indices: number[] = new Array(arrays.length).fill(0);
  const heap: HeapEntry[] = [];

  // Initialize heap with first element from each non-empty array
  for (let i = 0; i < arrays.length; i++) {
    if (arrays[i].length > 0) {
      const entry: HeapEntry = { arrayIdx: i, row: arrays[i][0] };
      heap.push(entry);
      indices[i] = 1;
    }
  }

  // Build heap
  for (let i = (heap.length >> 1) - 1; i >= 0; i--) {
    siftDown(heap, i, sortCol, asc);
  }

  const result: Row[] = [];

  while (heap.length > 0 && result.length < limit) {
    // Pop min
    const top = heap[0];
    result.push(top.row);

    const ai = top.arrayIdx;
    if (indices[ai] < arrays[ai].length) {
      heap[0] = { arrayIdx: ai, row: arrays[ai][indices[ai]] };
      indices[ai]++;
      siftDown(heap, 0, sortCol, asc);
    } else {
      // Remove top by replacing with last
      heap[0] = heap[heap.length - 1];
      heap.pop();
      if (heap.length > 0) {
        siftDown(heap, 0, sortCol, asc);
      }
    }
  }

  return result;
}

export function mergeQueryResults(
  partials: QueryResult[],
  query: QueryDescriptor,
): QueryResult {
  const totalBytesRead = partials.reduce((s, p) => s + p.bytesRead, 0);
  const totalPagesSkipped = partials.reduce((s, p) => s + p.pagesSkipped, 0);
  const maxDuration = partials.length > 0 ? partials.reduce((m, p) => p.durationMs > m ? p.durationMs : m, 0) : 0;
  const columns =
    partials.length > 0 ? partials[0].columns : query.projections;

  // Aggregation path
  if (query.aggregates && query.aggregates.length > 0) {
    const aggPartials = partials.map((p) => computePartialAgg(p.rows, query));
    const merged = mergePartialAggs(aggPartials);
    const rows = finalizePartialAgg(merged, query);
    return {
      rows,
      rowCount: rows.length,
      columns,
      bytesRead: totalBytesRead,
      pagesSkipped: totalPagesSkipped,
      durationMs: maxDuration,
    };
  }

  // Sort path: k-way merge (works with or without limit)
  if (query.sortColumn) {
    const rows = kWayMerge(
      partials.map((p) => p.rows),
      query.sortColumn,
      query.sortDirection ?? "asc",
      query.limit ?? Infinity,
    );
    return {
      rows,
      rowCount: rows.length,
      columns,
      bytesRead: totalBytesRead,
      pagesSkipped: totalPagesSkipped,
      durationMs: maxDuration,
    };
  }

  // Unsorted: limit-aware concat with early termination
  let rows: Row[];
  if (query.limit) {
    rows = [];
    for (const p of partials) {
      for (const row of p.rows) {
        rows.push(row);
        if (rows.length >= query.limit) break;
      }
      if (rows.length >= query.limit) break;
    }
  } else {
    rows = partials.flatMap((p) => p.rows);
  }

  return {
    rows,
    rowCount: rows.length,
    columns,
    bytesRead: totalBytesRead,
    pagesSkipped: totalPagesSkipped,
    durationMs: maxDuration,
  };
}
