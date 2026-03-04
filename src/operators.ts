/**
 * Streaming operator pipeline for QueryMode.
 *
 * Pull-based: the final operator pulls batches lazily from upstream.
 * Each operator processes one batch at a time — bounded memory.
 *
 * Pipeline: Scan → Filter → Aggregate|TopK|Sort|Limit → Project
 */

import type { ColumnMeta, PageInfo, Row } from "./types.js";
import type { QueryDescriptor } from "./client.js";
import type { WasmEngine } from "./wasm-engine.js";
import { canSkipPage, matchesFilter, decodePage } from "./decode.js";
import { decodeParquetColumnChunk } from "./parquet-decode.js";
import {
  computePartialAgg,
  mergePartialAggs,
  finalizePartialAgg,
  type PartialAgg,
} from "./partial-agg.js";

/** A batch of rows flowing through the pipeline. */
export type RowBatch = Row[];

/** Pull-based operator interface. */
export interface Operator {
  /** Pull the next batch of rows. Returns null when exhausted. */
  next(): Promise<RowBatch | null>;
  /** Release resources. */
  close(): Promise<void>;
}

// ---------------------------------------------------------------------------
// Fragment descriptor — everything needed to scan one data file
// ---------------------------------------------------------------------------

export interface FragmentSource {
  /** Columns to read (already filtered to needed set). */
  columns: ColumnMeta[];
  /** Read page data from the file. */
  readPage(col: ColumnMeta, page: PageInfo): Promise<ArrayBuffer>;
}

// ---------------------------------------------------------------------------
// ScanOperator — reads pages from fragments, decodes, yields batches
// ---------------------------------------------------------------------------

export class ScanOperator implements Operator {
  private fragments: FragmentSource[];
  private query: QueryDescriptor;
  private wasm: WasmEngine;
  private fragIdx = 0;
  private pageIdx = 0;
  pagesSkipped = 0;
  bytesRead = 0;

  constructor(fragments: FragmentSource[], query: QueryDescriptor, wasm: WasmEngine) {
    this.fragments = fragments;
    this.query = query;
    this.wasm = wasm;
  }

  async next(): Promise<RowBatch | null> {
    while (this.fragIdx < this.fragments.length) {
      const frag = this.fragments[this.fragIdx];
      const firstCol = frag.columns[0];
      if (!firstCol) { this.fragIdx++; this.pageIdx = 0; continue; }

      const totalPages = firstCol.pages.length;

      while (this.pageIdx < totalPages) {
        const pi = this.pageIdx;
        this.pageIdx++;

        // Page-level skip via min/max stats on first column
        const page = firstCol.pages[pi];
        if (!this.query.vectorSearch && canSkipPage(page, this.query.filters, firstCol.name)) {
          this.pagesSkipped += frag.columns.length;
          continue;
        }

        // Read + decode this page for all columns
        const pageInfoMap = new Map<string, { buf: ArrayBuffer; pageInfo: PageInfo }>();
        for (const col of frag.columns) {
          const colPage = col.pages[pi];
          if (!colPage) continue;
          if (!this.query.vectorSearch && canSkipPage(colPage, this.query.filters, col.name)) {
            this.pagesSkipped++;
            continue;
          }
          const buf = await frag.readPage(col, colPage);
          this.bytesRead += buf.byteLength;
          pageInfoMap.set(col.name, { buf, pageInfo: colPage });
        }

        // Decode columns for this single page
        const decoded = decodePageBatch(pageInfoMap, frag.columns, this.wasm);
        const firstDecoded = decoded.values().next().value;
        if (!firstDecoded || firstDecoded.length === 0) continue;
        const rowCount = firstDecoded.length;

        // Assemble rows
        const rows: Row[] = [];
        for (let i = 0; i < rowCount; i++) {
          const row: Row = {};
          for (const col of frag.columns) {
            const vals = decoded.get(col.name);
            row[col.name] = vals ? (vals[i] as Row[string]) : null;
          }
          rows.push(row);
        }
        return rows;
      }

      // Move to next fragment
      this.fragIdx++;
      this.pageIdx = 0;
    }

    return null;
  }

  async close(): Promise<void> { /* no-op */ }
}

// ---------------------------------------------------------------------------
// Shared page decode helper
// ---------------------------------------------------------------------------

type DecodedValue = number | bigint | string | boolean | Float32Array | null;

/** Decode one page per column given page buffer + metadata. */
function decodePageBatch(
  pageInfoMap: Map<string, { buf: ArrayBuffer; pageInfo: PageInfo }>,
  columns: ColumnMeta[],
  wasm: WasmEngine,
): Map<string, DecodedValue[]> {
  const result = new Map<string, DecodedValue[]>();
  for (const col of columns) {
    const entry = pageInfoMap.get(col.name);
    if (!entry) continue;
    const { buf, pageInfo: pi } = entry;

    if (col.dtype === "fixed_size_list") {
      const dim = col.listDimension ?? 0;
      if (dim === 0) continue;
      const vecs: Float32Array[] = [];
      const view = new DataView(buf);
      const numRows = Math.floor((buf.byteLength >> 2) / dim);
      for (let r = 0; r < numRows; r++) {
        const vec = new Float32Array(dim);
        for (let d = 0; d < dim; d++) vec[d] = view.getFloat32((r * dim + d) * 4, true);
        vecs.push(vec);
      }
      result.set(col.name, vecs);
    } else {
      const decoded = pi.encoding
        ? decodeParquetColumnChunk(buf, pi.encoding, col.dtype, pi.rowCount, wasm)
        : decodePage(buf, col.dtype, pi.nullCount ?? 0, pi.rowCount ?? 0, pi.dataOffsetInPage);
      result.set(col.name, decoded as DecodedValue[]);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// FilterOperator — applies row-level filters
// ---------------------------------------------------------------------------

export class FilterOperator implements Operator {
  private upstream: Operator;
  private filters: QueryDescriptor["filters"];

  constructor(upstream: Operator, filters: QueryDescriptor["filters"]) {
    this.upstream = upstream;
    this.filters = filters;
  }

  async next(): Promise<RowBatch | null> {
    while (true) {
      const batch = await this.upstream.next();
      if (!batch) return null;

      const filtered: Row[] = [];
      for (const row of batch) {
        if (this.filters.every(f => {
          const v = row[f.column];
          return v !== null && matchesFilter(v, f);
        })) {
          filtered.push(row);
        }
      }
      if (filtered.length > 0) return filtered;
      // Empty batch after filtering — pull next
    }
  }

  async close(): Promise<void> {
    await this.upstream.close();
  }
}

// ---------------------------------------------------------------------------
// LimitOperator — short-circuits after N rows
// ---------------------------------------------------------------------------

export class LimitOperator implements Operator {
  private upstream: Operator;
  private remaining: number;
  private offset: number;
  private skipped = 0;

  constructor(upstream: Operator, limit: number, offset = 0) {
    this.upstream = upstream;
    this.remaining = limit;
    this.offset = offset;
  }

  async next(): Promise<RowBatch | null> {
    if (this.remaining <= 0) return null;

    while (true) {
      const batch = await this.upstream.next();
      if (!batch) return null;

      let rows = batch;

      // Handle offset: skip rows
      if (this.skipped < this.offset) {
        const toSkip = this.offset - this.skipped;
        if (toSkip >= rows.length) {
          this.skipped += rows.length;
          continue;
        }
        rows = rows.slice(toSkip);
        this.skipped = this.offset;
      }

      // Apply limit
      if (rows.length > this.remaining) {
        rows = rows.slice(0, this.remaining);
      }
      this.remaining -= rows.length;
      return rows;
    }
  }

  async close(): Promise<void> {
    await this.upstream.close();
  }
}

// ---------------------------------------------------------------------------
// ProjectOperator — strips non-projected columns
// ---------------------------------------------------------------------------

export class ProjectOperator implements Operator {
  private upstream: Operator;
  private keep: Set<string>;

  constructor(upstream: Operator, columns: string[]) {
    this.upstream = upstream;
    this.keep = new Set(columns);
  }

  async next(): Promise<RowBatch | null> {
    const batch = await this.upstream.next();
    if (!batch) return null;

    for (const row of batch) {
      for (const key of Object.keys(row)) {
        if (!this.keep.has(key)) delete row[key];
      }
    }
    return batch;
  }

  async close(): Promise<void> {
    await this.upstream.close();
  }
}

// ---------------------------------------------------------------------------
// AggregateOperator — streaming partial aggregation, O(groups) memory
// ---------------------------------------------------------------------------

export class AggregateOperator implements Operator {
  private upstream: Operator;
  private query: QueryDescriptor;
  private consumed = false;

  constructor(upstream: Operator, query: QueryDescriptor) {
    this.upstream = upstream;
    this.query = query;
  }

  async next(): Promise<RowBatch | null> {
    if (this.consumed) return null;
    this.consumed = true;

    // Stream all batches through partial aggregation
    const partials: PartialAgg[] = [];
    while (true) {
      const batch = await this.upstream.next();
      if (!batch) break;
      partials.push(computePartialAgg(batch, this.query));
    }

    if (partials.length === 0) {
      return finalizePartialAgg({ states: [] }, this.query);
    }

    const merged = partials.length === 1 ? partials[0] : mergePartialAggs(partials);
    return finalizePartialAgg(merged, this.query);
  }

  async close(): Promise<void> {
    await this.upstream.close();
  }
}

// ---------------------------------------------------------------------------
// TopKOperator — ORDER BY + LIMIT using min-heap, O(K) memory
// ---------------------------------------------------------------------------

export class TopKOperator implements Operator {
  private upstream: Operator;
  private k: number;
  private col: string;
  private desc: boolean;
  private offset: number;
  private consumed = false;

  constructor(upstream: Operator, sortColumn: string, desc: boolean, k: number, offset = 0) {
    this.upstream = upstream;
    this.col = sortColumn;
    this.desc = desc;
    this.k = k + offset; // need offset+limit for correct slicing
    this.offset = offset;
  }

  async next(): Promise<RowBatch | null> {
    if (this.consumed) return null;
    this.consumed = true;

    const heap: Row[] = [];
    const col = this.col;
    const desc = this.desc;
    const k = this.k;

    const cmp = (a: Row, b: Row): number => {
      const av = a[col], bv = b[col];
      if (av === null && bv === null) return 0;
      if (av === null) return -1;
      if (bv === null) return 1;
      const c = av < bv ? -1 : av > bv ? 1 : 0;
      return desc ? -c : c;
    };

    const shouldReplace = (row: Row): boolean => {
      const nv = row[col], rv = heap[0][col];
      if (nv === null) return false;
      if (rv === null) return true;
      return desc ? nv > rv : nv < rv;
    };

    const siftDown = (arr: Row[], i: number): void => {
      while (true) {
        let t = i;
        const l = 2 * i + 1, r = 2 * i + 2;
        if (l < arr.length && cmp(arr[l], arr[t]) > 0) t = l;
        if (r < arr.length && cmp(arr[r], arr[t]) > 0) t = r;
        if (t === i) break;
        [arr[i], arr[t]] = [arr[t], arr[i]];
        i = t;
      }
    };

    const siftUp = (i: number): void => {
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (cmp(heap[i], heap[p]) > 0) { [heap[i], heap[p]] = [heap[p], heap[i]]; i = p; }
        else break;
      }
    };

    // Stream all batches through the heap
    while (true) {
      const batch = await this.upstream.next();
      if (!batch) break;
      for (const row of batch) {
        if (heap.length < k) { heap.push(row); siftUp(heap.length - 1); }
        else if (heap.length > 0 && shouldReplace(row)) { heap[0] = row; siftDown(heap, 0); }
      }
    }

    // Extract sorted from heap
    const result: Row[] = [];
    const copy = [...heap];
    while (copy.length > 0) {
      result.push(copy[0]);
      copy[0] = copy[copy.length - 1];
      copy.pop();
      if (copy.length > 0) siftDown(copy, 0);
    }
    result.reverse();

    // Apply offset
    return result.slice(this.offset);
  }

  async close(): Promise<void> {
    await this.upstream.close();
  }
}

// ---------------------------------------------------------------------------
// InMemorySortOperator — ORDER BY without LIMIT (full sort in memory)
// ---------------------------------------------------------------------------

export class InMemorySortOperator implements Operator {
  private upstream: Operator;
  private col: string;
  private desc: boolean;
  private offset: number;
  private consumed = false;

  constructor(upstream: Operator, sortColumn: string, desc: boolean, offset = 0) {
    this.upstream = upstream;
    this.col = sortColumn;
    this.desc = desc;
    this.offset = offset;
  }

  async next(): Promise<RowBatch | null> {
    if (this.consumed) return null;
    this.consumed = true;

    // Collect all rows (required for full sort without limit)
    const allRows: Row[] = [];
    while (true) {
      const batch = await this.upstream.next();
      if (!batch) break;
      for (const row of batch) allRows.push(row);
    }

    const col = this.col;
    const dir = this.desc ? -1 : 1;
    allRows.sort((a, b) => {
      const av = a[col], bv = b[col];
      if (av === null && bv === null) return 0;
      if (av === null) return 1;
      if (bv === null) return -1;
      return av < bv ? -dir : av > bv ? dir : 0;
    });

    return this.offset > 0 ? allRows.slice(this.offset) : allRows;
  }

  async close(): Promise<void> {
    await this.upstream.close();
  }
}

// ---------------------------------------------------------------------------
// WasmAggregateOperator — SIMD aggregates on raw column buffers (no row creation)
// ---------------------------------------------------------------------------

/**
 * Check if a query can use the WASM SIMD fast path for aggregation.
 * Requirements:
 * - No groupBy (groupBy requires row-level key extraction)
 * - No filters (filter requires row-level evaluation; page skip is fine)
 * - All aggregate columns are numeric (float64 or int64)
 * - Only sum/min/max/avg/count functions
 */
export function canUseWasmAggregate(query: QueryDescriptor, columns: ColumnMeta[]): boolean {
  if (!query.aggregates || query.aggregates.length === 0) return false;
  if (query.groupBy && query.groupBy.length > 0) return false;
  if (query.filters.length > 0) return false;

  const colMap = new Map(columns.map(c => [c.name, c]));
  for (const agg of query.aggregates) {
    if (agg.column === "*") continue; // count(*) is always fine
    const col = colMap.get(agg.column);
    if (!col) return false;
    // Only float64 and int64 have WASM SIMD paths
    if (col.dtype !== "float64" && col.dtype !== "int64") return false;
    // Only Parquet pages without custom encoding can use raw buffer path
    if (col.pages.some(p => p.encoding)) return false;
    // Skip nullable columns (need bitmap handling)
    if (col.nullCount > 0) return false;
  }
  return true;
}

/**
 * WASM SIMD aggregate operator. Reads raw page buffers directly and feeds
 * them to WASM SIMD sum/min/max/avg without creating Row objects.
 * O(1) memory regardless of table size.
 */
export class WasmAggregateOperator implements Operator {
  private fragments: FragmentSource[];
  private query: QueryDescriptor;
  private wasm: WasmEngine;
  private consumed = false;
  bytesRead = 0;
  pagesSkipped = 0;

  constructor(fragments: FragmentSource[], query: QueryDescriptor, wasm: WasmEngine) {
    this.fragments = fragments;
    this.query = query;
    this.wasm = wasm;
  }

  async next(): Promise<RowBatch | null> {
    if (this.consumed) return null;
    this.consumed = true;

    const aggregates = this.query.aggregates ?? [];
    // Accumulator per aggregate
    const acc: { sum: number; count: number; min: number; max: number }[] =
      aggregates.map(() => ({ sum: 0, count: 0, min: Infinity, max: -Infinity }));

    for (const frag of this.fragments) {
      const colMap = new Map(frag.columns.map(c => [c.name, c]));

      for (let ai = 0; ai < aggregates.length; ai++) {
        const agg = aggregates[ai];
        if (agg.column === "*") {
          // count(*): just sum row counts from page metadata
          const firstCol = frag.columns[0];
          if (firstCol) {
            for (const page of firstCol.pages) {
              acc[ai].count += page.rowCount;
            }
          }
          continue;
        }

        const col = colMap.get(agg.column);
        if (!col) continue;

        for (const page of col.pages) {
          if (canSkipPage(page, this.query.filters, col.name)) {
            this.pagesSkipped++;
            continue;
          }

          const buf = await frag.readPage(col, page);
          this.bytesRead += buf.byteLength;

          if (col.dtype === "float64") {
            const count = buf.byteLength >> 3;
            acc[ai].count += count;
            if (agg.fn === "sum" || agg.fn === "avg") {
              acc[ai].sum += this.wasm.sumFloat64(buf);
            }
            if (agg.fn === "min") {
              const v = this.wasm.minFloat64(buf);
              if (v < acc[ai].min) acc[ai].min = v;
            }
            if (agg.fn === "max") {
              const v = this.wasm.maxFloat64(buf);
              if (v > acc[ai].max) acc[ai].max = v;
            }
          } else if (col.dtype === "int64") {
            const count = buf.byteLength >> 3;
            acc[ai].count += count;
            // For int64 we need to read values as BigInt64Array
            const view = new BigInt64Array(buf);
            for (const v of view) {
              const n = Number(v);
              acc[ai].sum += n;
              if (n < acc[ai].min) acc[ai].min = n;
              if (n > acc[ai].max) acc[ai].max = n;
            }
          }
        }
      }
    }

    // Build result row
    const row: Row = {};
    for (let i = 0; i < aggregates.length; i++) {
      const agg = aggregates[i];
      const alias = agg.alias ?? `${agg.fn}_${agg.column}`;
      switch (agg.fn) {
        case "sum": row[alias] = acc[i].sum; break;
        case "avg": row[alias] = acc[i].count === 0 ? 0 : acc[i].sum / acc[i].count; break;
        case "min": row[alias] = acc[i].min; break;
        case "max": row[alias] = acc[i].max; break;
        case "count": row[alias] = acc[i].count; break;
      }
    }

    return [row];
  }

  async close(): Promise<void> { /* no-op */ }
}

// ---------------------------------------------------------------------------
// HashJoinOperator — build hash map from right side, probe with left batches
// ---------------------------------------------------------------------------

export class HashJoinOperator implements Operator {
  private left: Operator;
  private right: Operator;
  private leftKey: string;
  private rightKey: string;
  private joinType: "inner" | "left";
  private hashMap: Map<string, Row[]> | null = null;

  constructor(
    left: Operator,
    right: Operator,
    leftKey: string,
    rightKey: string,
    joinType: "inner" | "left" = "inner",
  ) {
    this.left = left;
    this.right = right;
    this.leftKey = leftKey;
    this.rightKey = rightKey;
    this.joinType = joinType;
  }

  private async buildHashMap(): Promise<Map<string, Row[]>> {
    if (this.hashMap) return this.hashMap;
    this.hashMap = new Map<string, Row[]>();

    while (true) {
      const batch = await this.right.next();
      if (!batch) break;
      for (const row of batch) {
        const key = this.toJoinKey(row[this.rightKey]);
        const bucket = this.hashMap.get(key);
        if (bucket) bucket.push(row);
        else this.hashMap.set(key, [row]);
      }
    }

    return this.hashMap;
  }

  private toJoinKey(val: Row[string]): string {
    if (val === null) return "__null__";
    if (typeof val === "bigint") return val.toString();
    return String(val);
  }

  async next(): Promise<RowBatch | null> {
    const map = await this.buildHashMap();

    while (true) {
      const batch = await this.left.next();
      if (!batch) return null;

      const result: Row[] = [];
      for (const leftRow of batch) {
        const key = this.toJoinKey(leftRow[this.leftKey]);
        const rightRows = map.get(key);

        if (rightRows) {
          for (const rightRow of rightRows) {
            const merged: Row = { ...leftRow };
            for (const k in rightRow) {
              if (k === this.rightKey) continue; // skip duplicate join key
              // Prefix right-side columns with right table name if they conflict
              const outKey = k in merged ? `right_${k}` : k;
              merged[outKey] = rightRow[k];
            }
            result.push(merged);
          }
        } else if (this.joinType === "left") {
          result.push({ ...leftRow });
        }
      }

      if (result.length > 0) return result;
    }
  }

  async close(): Promise<void> {
    this.hashMap = null;
    await this.left.close();
    await this.right.close();
  }
}

// ---------------------------------------------------------------------------
// ExternalSortOperator — ORDER BY without LIMIT, spills to temp files
// ---------------------------------------------------------------------------

/** Default memory budget for external sort: 256MB */
export const DEFAULT_MEMORY_BUDGET = 256 * 1024 * 1024;

/** Rough estimate of a row's memory footprint in bytes. */
function estimateRowSize(row: Row): number {
  let size = 64; // object overhead
  for (const key in row) {
    const val = row[key];
    if (typeof val === "string") size += 40 + val.length * 2;
    else if (val instanceof Float32Array) size += 40 + val.byteLength;
    else size += 16;
  }
  return size;
}

export class ExternalSortOperator implements Operator {
  private upstream: Operator;
  private col: string;
  private desc: boolean;
  private offset: number;
  private memoryBudget: number;
  private consumed = false;
  private runFiles: string[] = [];
  private cleanup: (() => Promise<void>) | null = null;

  constructor(
    upstream: Operator,
    sortColumn: string,
    desc: boolean,
    offset = 0,
    memoryBudget = DEFAULT_MEMORY_BUDGET,
  ) {
    this.upstream = upstream;
    this.col = sortColumn;
    this.desc = desc;
    this.offset = offset;
    this.memoryBudget = memoryBudget;
  }

  async next(): Promise<RowBatch | null> {
    if (this.consumed) return null;
    this.consumed = true;

    const col = this.col;
    const dir = this.desc ? -1 : 1;
    const compareFn = (a: Row, b: Row): number => {
      const av = a[col], bv = b[col];
      if (av === null && bv === null) return 0;
      if (av === null) return 1;
      if (bv === null) return -1;
      return av < bv ? -dir : av > bv ? dir : 0;
    };

    // Phase 1: Generate sorted runs
    const fs = await import("node:fs/promises");
    const os = await import("node:os");
    const path = await import("node:path");
    const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "qm-sort-"));

    let currentRun: Row[] = [];
    let currentRunBytes = 0;

    const flushRun = async (): Promise<void> => {
      if (currentRun.length === 0) return;
      currentRun.sort(compareFn);
      const runPath = path.join(tmpDir, `run_${this.runFiles.length}.ndjson`);
      const lines = currentRun.map(row => JSON.stringify(row, bigIntJsonReplacer));
      await fs.writeFile(runPath, lines.join("\n") + "\n");
      this.runFiles.push(runPath);
      currentRun = [];
      currentRunBytes = 0;
    };

    while (true) {
      const batch = await this.upstream.next();
      if (!batch) break;
      for (const row of batch) {
        const rowSize = estimateRowSize(row);
        if (currentRunBytes + rowSize > this.memoryBudget && currentRun.length > 0) {
          await flushRun();
        }
        currentRun.push(row);
        currentRunBytes += rowSize;
      }
    }

    // If everything fit in one run, no temp files needed
    if (this.runFiles.length === 0) {
      currentRun.sort(compareFn);
      // Cleanup empty temp dir
      await fs.rmdir(tmpDir).catch(() => {});
      return this.offset > 0 ? currentRun.slice(this.offset) : currentRun;
    }

    // Flush final run
    await flushRun();

    // Phase 2: K-way merge using min-heap
    const result = await this.kWayMerge(compareFn);

    // Schedule cleanup
    this.cleanup = async () => {
      for (const f of this.runFiles) await fs.unlink(f).catch(() => {});
      await fs.rmdir(tmpDir).catch(() => {});
    };
    await this.cleanup();
    this.cleanup = null;

    return this.offset > 0 ? result.slice(this.offset) : result;
  }

  private async kWayMerge(compareFn: (a: Row, b: Row) => number): Promise<Row[]> {
    const fs = await import("node:fs/promises");
    const readline = await import("node:readline");
    const nodeFs = await import("node:fs");

    // Open all run files as line readers
    type RunReader = { rl: ReturnType<typeof readline.createInterface>; lines: AsyncIterator<string>; current: Row | null; stream: ReturnType<typeof nodeFs.createReadStream> };
    const readers: RunReader[] = [];

    for (const filePath of this.runFiles) {
      const stream = nodeFs.createReadStream(filePath);
      const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
      const lines = rl[Symbol.asyncIterator]();
      const first = await lines.next();
      const current = first.done ? null : parseNdjsonRow(first.value);
      readers.push({ rl, lines, current, stream });
    }

    // Min-heap of reader indices
    const heap: number[] = [];
    for (let i = 0; i < readers.length; i++) {
      if (readers[i].current !== null) {
        heap.push(i);
      }
    }

    // Build heap
    const heapCmp = (a: number, b: number): number => {
      return compareFn(readers[a].current!, readers[b].current!);
    };

    const siftDown = (i: number): void => {
      while (true) {
        let t = i;
        const l = 2 * i + 1, r = 2 * i + 2;
        if (l < heap.length && heapCmp(heap[l], heap[t]) < 0) t = l;
        if (r < heap.length && heapCmp(heap[r], heap[t]) < 0) t = r;
        if (t === i) break;
        [heap[i], heap[t]] = [heap[t], heap[i]];
        i = t;
      }
    };

    // Build min-heap
    for (let i = Math.floor(heap.length / 2) - 1; i >= 0; i--) siftDown(i);

    const result: Row[] = [];
    while (heap.length > 0) {
      const minIdx = heap[0];
      result.push(readers[minIdx].current!);

      // Advance the reader
      const next = await readers[minIdx].lines.next();
      if (next.done) {
        readers[minIdx].current = null;
        // Remove from heap: swap with last, pop, sift down
        heap[0] = heap[heap.length - 1];
        heap.pop();
        if (heap.length > 0) siftDown(0);
      } else {
        readers[minIdx].current = parseNdjsonRow(next.value);
        siftDown(0);
      }
    }

    // Close all readers
    for (const r of readers) {
      r.rl.close();
      r.stream.destroy();
    }

    return result;
  }

  async close(): Promise<void> {
    if (this.cleanup) await this.cleanup();
    await this.upstream.close();
  }
}

/** JSON replacer for bigint values in NDJSON runs. */
function bigIntJsonReplacer(_key: string, value: unknown): unknown {
  return typeof value === "bigint" ? `__bigint__${value.toString()}` : value;
}

/** Parse a single NDJSON line, restoring bigint values. */
function parseNdjsonRow(line: string): Row {
  return JSON.parse(line, (_key, value) => {
    if (typeof value === "string" && value.startsWith("__bigint__")) {
      return BigInt(value.slice(10));
    }
    return value;
  }) as Row;
}

// ---------------------------------------------------------------------------
// Pipeline builder
// ---------------------------------------------------------------------------

export interface PipelineResult {
  pipeline: Operator;
  scan: ScanOperator | null;
  /** If WASM aggregate fast path is used, stats come from here instead of scan. */
  wasmAgg?: WasmAggregateOperator;
}

export interface PipelineOptions {
  /** Memory budget in bytes for external sort (default 256MB). */
  memoryBudgetBytes?: number;
}

/**
 * Build an operator pipeline from a query descriptor and fragment sources.
 * Returns the terminal operator (pull from it) and the scan operator (for stats).
 */
export function buildPipeline(
  fragments: FragmentSource[],
  query: QueryDescriptor,
  wasm: WasmEngine,
  outputColumns: string[],
  options?: PipelineOptions,
): PipelineResult {
  // WASM SIMD fast path: pure numeric aggregates without groupBy or filters
  const allColumns = fragments.flatMap(f => f.columns);
  if (canUseWasmAggregate(query, allColumns)) {
    const wasmAgg = new WasmAggregateOperator(fragments, query, wasm);
    return { pipeline: wasmAgg, scan: null, wasmAgg };
  }

  const scan = new ScanOperator(fragments, query, wasm);
  let pipeline: Operator = scan;
  const memBudget = options?.memoryBudgetBytes ?? DEFAULT_MEMORY_BUDGET;

  // Filter
  if (query.filters.length > 0) {
    pipeline = new FilterOperator(pipeline, query.filters);
  }

  const hasAgg = query.aggregates && query.aggregates.length > 0;

  if (hasAgg) {
    // Aggregate: consumes all filtered rows, produces aggregate output
    pipeline = new AggregateOperator(pipeline, query);

    // Sort/limit on aggregate output (aggregate output is small — always in-memory)
    if (query.sortColumn && query.limit !== undefined) {
      pipeline = new TopKOperator(
        pipeline, query.sortColumn, query.sortDirection === "desc",
        query.limit, query.offset ?? 0,
      );
    } else if (query.sortColumn) {
      pipeline = new InMemorySortOperator(
        pipeline, query.sortColumn, query.sortDirection === "desc",
        query.offset ?? 0,
      );
    } else if (query.offset || query.limit !== undefined) {
      pipeline = new LimitOperator(pipeline, query.limit ?? Infinity, query.offset ?? 0);
    }
  } else if (query.sortColumn) {
    // Sort path
    if (query.limit !== undefined) {
      pipeline = new TopKOperator(
        pipeline, query.sortColumn, query.sortDirection === "desc",
        query.limit, query.offset ?? 0,
      );
    } else {
      // Full sort without limit — use external sort to handle TB+ data
      pipeline = new ExternalSortOperator(
        pipeline, query.sortColumn, query.sortDirection === "desc",
        query.offset ?? 0, memBudget,
      );
    }
  } else if (query.offset || query.limit !== undefined) {
    // No sort — streaming limit with offset
    pipeline = new LimitOperator(pipeline, query.limit ?? Infinity, query.offset ?? 0);
  }

  // Project — strip extra columns at the end
  if (outputColumns.length > 0) {
    pipeline = new ProjectOperator(pipeline, outputColumns);
  }

  return { pipeline, scan };
}

/**
 * Drain all rows from an operator pipeline.
 */
export async function drainPipeline(pipeline: Operator): Promise<Row[]> {
  const rows: Row[] = [];
  while (true) {
    const batch = await pipeline.next();
    if (!batch) break;
    for (const row of batch) rows.push(row);
  }
  await pipeline.close();
  return rows;
}
