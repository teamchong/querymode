import type { Row, QueryResult } from "./types.js";
import type { QueryDescriptor } from "./client.js";
import {
  computePartialAgg,
  computePartialAggQMCB,
  mergePartialAggs,
  finalizePartialAgg,
} from "./partial-agg.js";
import {
  decodeColumnarBatch,
  columnarBatchToRows,
  concatColumnarBatches,
  columnarKWayMerge,
  sliceColumnarBatch,
  encodeColumnarBatch,
  type ColumnarBatch,
} from "./columnar.js";

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
      top.row = arrays[ai][indices[ai]];
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

/** Decode columnarData on a partial to ColumnarBatch. Returns null if no columnar data. */
function partialToColumnarBatch(p: QueryResult): ColumnarBatch | null {
  if (p.columnarData) {
    const batch = decodeColumnarBatch(p.columnarData);
    p.columnarData = undefined; // free binary after decode
    return batch;
  }
  return null;
}

/** Ensure partial has rows (decode columnar if needed). Mutates partial. */
function ensureRows(p: QueryResult): void {
  if (p.columnarData && p.rows.length === 0) {
    const batch = decodeColumnarBatch(p.columnarData);
    if (batch) p.rows = columnarBatchToRows(batch);
    p.columnarData = undefined;
  }
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

  const baseResult = {
    bytesRead: totalBytesRead,
    pagesSkipped: totalPagesSkipped,
    durationMs: maxDuration,
  };

  // Aggregation path — columnar when possible, Row[] fallback
  if (query.aggregates && query.aggregates.length > 0) {
    const aggPartials = partials.map((p) => {
      if (p.columnarData) {
        const batch = partialToColumnarBatch(p);
        if (batch) return computePartialAggQMCB(batch, query);
      }
      return computePartialAgg(p.rows, query);
    });
    const merged = mergePartialAggs(aggPartials);
    const rows = finalizePartialAgg(merged, query);
    return { rows, rowCount: rows.length, columns, ...baseResult };
  }

  // Check if all partials are columnar
  const allColumnar = partials.every(p => p.columnarData || p.rows.length === 0);

  // Sort path: columnar k-way merge
  if (query.sortColumn) {
    const off = query.offset ?? 0;
    const effectiveLimit = (query.limit ?? Infinity) + off;

    if (allColumnar) {
      const batches: ColumnarBatch[] = [];
      for (const p of partials) {
        const batch = partialToColumnarBatch(p);
        if (batch && batch.rowCount > 0) batches.push(batch);
      }
      if (batches.length > 0) {
        let merged = columnarKWayMerge(batches, query.sortColumn, query.sortDirection ?? "asc", effectiveLimit);
        if (off > 0) merged = sliceColumnarBatch(merged, off);
        const columnarData = encodeColumnarBatch(merged);
        return { rows: [], columnarData, rowCount: merged.rowCount, columns, ...baseResult };
      }
      return { rows: [], rowCount: 0, columns, ...baseResult };
    }

    // Fallback: mixed or row-based partials
    for (const p of partials) ensureRows(p);
    let rows = kWayMerge(
      partials.map((p) => p.rows),
      query.sortColumn,
      query.sortDirection ?? "asc",
      effectiveLimit,
    );
    if (off > 0) rows = rows.slice(off);
    return { rows, rowCount: rows.length, columns, ...baseResult };
  }

  // Unsorted: columnar concat with offset + limit
  if (allColumnar) {
    const batches: ColumnarBatch[] = [];
    for (const p of partials) {
      const batch = partialToColumnarBatch(p);
      if (batch && batch.rowCount > 0) batches.push(batch);
    }
    if (batches.length > 0) {
      let merged = concatColumnarBatches(batches)!;
      const off = query.offset ?? 0;
      if (off > 0 || query.limit !== undefined) {
        merged = sliceColumnarBatch(merged, off, query.limit);
      }
      const columnarData = encodeColumnarBatch(merged);
      return { rows: [], columnarData, rowCount: merged.rowCount, columns, ...baseResult };
    }
    return { rows: [], rowCount: 0, columns, ...baseResult };
  }

  // Fallback: mixed or row-based
  for (const p of partials) ensureRows(p);
  let rows: Row[];
  const allRows = partials.flatMap((p) => p.rows);
  const off = query.offset ?? 0;
  if (off > 0 || query.limit !== undefined) {
    rows = allRows.slice(off, query.limit !== undefined ? off + query.limit : undefined);
  } else {
    rows = allRows;
  }

  return { rows, rowCount: rows.length, columns, ...baseResult };
}
