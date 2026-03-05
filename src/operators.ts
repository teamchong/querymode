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

    // Create new projected row objects — no delete, no V8 hidden class deopt
    return batch.map(row => {
      const out: Row = {};
      for (const k of this.keep) if (k in row) out[k] = row[k];
      return out;
    });
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

    // Incrementally merge partials — O(groups) memory, not O(batches × groups)
    let merged: PartialAgg | null = null;
    while (true) {
      const batch = await this.upstream.next();
      if (!batch) break;
      const partial = computePartialAgg(batch, this.query);
      merged = merged ? mergePartialAggs([merged, partial]) : partial;
    }

    if (!merged) {
      return finalizePartialAgg({ states: [] }, this.query);
    }
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
  private memoryBudget: number;
  private spill: import("./r2-spill.js").SpillBackend | null;

  // In-memory build phase state
  private hashMap: Map<string, Row[]> | null = null;
  private buildExceeded = false;
  private buildSizeBytes = 0;

  // Partitioned spill state (Grace hash join)
  private partitionCount = 0;
  private leftPartitionIds: string[] = [];
  private rightPartitionIds: string[] = [];
  private currentPartition = 0;
  private partitionedResult: Row[] | null = null;
  private partitionedDrained = false;

  constructor(
    left: Operator,
    right: Operator,
    leftKey: string,
    rightKey: string,
    joinType: "inner" | "left" = "inner",
    memoryBudget = DEFAULT_MEMORY_BUDGET,
    spill?: import("./r2-spill.js").SpillBackend,
  ) {
    this.left = left;
    this.right = right;
    this.leftKey = leftKey;
    this.rightKey = rightKey;
    this.joinType = joinType;
    this.memoryBudget = memoryBudget;
    this.spill = spill ?? null;
  }

  private toJoinKey(val: Row[string]): string {
    if (val === null) return "__null__";
    if (typeof val === "bigint") return val.toString();
    return String(val);
  }

  private hashPartition(val: Row[string], numPartitions: number): number {
    const key = this.toJoinKey(val);
    let h = 0;
    for (let i = 0; i < key.length; i++) {
      h = ((h << 5) - h + key.charCodeAt(i)) | 0;
    }
    return ((h % numPartitions) + numPartitions) % numPartitions;
  }

  private mergeRow(leftRow: Row, rightRow: Row): Row {
    const merged: Row = { ...leftRow };
    for (const k in rightRow) {
      if (k === this.rightKey) continue;
      const outKey = k in merged ? `right_${k}` : k;
      merged[outKey] = rightRow[k];
    }
    return merged;
  }

  /** Try to build hash map in memory. If build side exceeds budget, switch to partition mode. */
  private async buildOrPartition(): Promise<void> {
    if (this.hashMap || this.partitionCount > 0) return;

    // Sample first batch to estimate whether right side fits in memory.
    // If estimated total exceeds budget, go straight to partitioned path
    // instead of accumulating rows we'll have to re-spill.
    const firstBatch = await this.right.next();
    if (!firstBatch) {
      // Empty right side — build empty hash map
      this.hashMap = new Map<string, Row[]>();
      return;
    }

    let batchSizeBytes = 0;
    for (const row of firstBatch) batchSizeBytes += estimateRowSize(row);
    const avgRowSize = batchSizeBytes / firstBatch.length;

    // Heuristic: if first batch already exceeds half the budget, go partitioned
    const goPartitioned = this.spill && batchSizeBytes > this.memoryBudget / 2;

    if (!goPartitioned) {
      // Optimistic in-memory path: consume right side, tracking memory
      const inMemoryRows: Row[] = [...firstBatch];
      this.buildSizeBytes = batchSizeBytes;
      let exceeds = false;

      while (true) {
        const batch = await this.right.next();
        if (!batch) break;
        for (const row of batch) {
          const rowSize = estimateRowSize(row);
          this.buildSizeBytes += rowSize;
          inMemoryRows.push(row);

          if (this.spill && this.buildSizeBytes > this.memoryBudget) {
            exceeds = true;
            break;
          }
        }
        if (exceeds) break;
      }

      if (!exceeds) {
        // Fits in memory — build hash map directly
        this.hashMap = new Map<string, Row[]>();
        for (const row of inMemoryRows) {
          const key = this.toJoinKey(row[this.rightKey]);
          const bucket = this.hashMap.get(key);
          if (bucket) bucket.push(row);
          else this.hashMap.set(key, [row]);
        }
        return;
      }

      // Fell through: exceeded budget mid-stream, partition what we have
      this.buildExceeded = true;
      this.partitionCount = Math.max(4, Math.ceil(this.buildSizeBytes / (this.memoryBudget / 2)));
      await this.partitionRightRows(inMemoryRows);
      inMemoryRows.length = 0;

      // Continue consuming remaining right-side rows
      await this.consumeRemainingRight();
      return;
    }

    // Proactive partition path: first batch signals right side is large
    this.buildExceeded = true;
    this.buildSizeBytes = batchSizeBytes;
    // Estimate partition count from first batch size × expected batch count
    // Use conservative estimate: assume at least 4× more data coming
    const estimatedTotal = batchSizeBytes * 4;
    this.partitionCount = Math.max(4, Math.ceil(estimatedTotal / (this.memoryBudget / 2)));

    await this.partitionRightRows(firstBatch);
    await this.consumeRemainingRight();
  }

  /** Partition an array of right-side rows into spill buckets. */
  private async partitionRightRows(rows: Row[]): Promise<void> {
    const bucketBudget = Math.floor(this.memoryBudget / this.partitionCount);
    const rightBuckets: Row[][] = Array.from({ length: this.partitionCount }, () => []);
    const rightBucketBytes: number[] = new Array(this.partitionCount).fill(0);
    if (!this.rightPartitionIds.length) {
      this.rightPartitionIds = new Array(this.partitionCount).fill("");
    }

    const flushBucket = async (bi: number): Promise<void> => {
      if (rightBuckets[bi].length === 0) return;
      const spillId = await this.spill!.writeRun(rightBuckets[bi]);
      this.rightPartitionIds[bi] = this.rightPartitionIds[bi]
        ? this.rightPartitionIds[bi] + "|" + spillId
        : spillId;
      rightBuckets[bi] = [];
      rightBucketBytes[bi] = 0;
    };

    for (const row of rows) {
      const bi = this.hashPartition(row[this.rightKey], this.partitionCount);
      const rowSize = estimateRowSize(row);
      if (rightBucketBytes[bi] + rowSize > bucketBudget && rightBuckets[bi].length > 0) {
        await flushBucket(bi);
      }
      rightBuckets[bi].push(row);
      rightBucketBytes[bi] += rowSize;
    }

    for (let bi = 0; bi < this.partitionCount; bi++) await flushBucket(bi);
  }

  /** Consume remaining right-side batches into partitions. */
  private async consumeRemainingRight(): Promise<void> {
    const bucketBudget = Math.floor(this.memoryBudget / this.partitionCount);
    const rightBuckets: Row[][] = Array.from({ length: this.partitionCount }, () => []);
    const rightBucketBytes: number[] = new Array(this.partitionCount).fill(0);

    const flushBucket = async (bi: number): Promise<void> => {
      if (rightBuckets[bi].length === 0) return;
      const spillId = await this.spill!.writeRun(rightBuckets[bi]);
      this.rightPartitionIds[bi] = this.rightPartitionIds[bi]
        ? this.rightPartitionIds[bi] + "|" + spillId
        : spillId;
      rightBuckets[bi] = [];
      rightBucketBytes[bi] = 0;
    };

    while (true) {
      const batch = await this.right.next();
      if (!batch) break;
      for (const row of batch) {
        const bi = this.hashPartition(row[this.rightKey], this.partitionCount);
        const rowSize = estimateRowSize(row);
        this.buildSizeBytes += rowSize;
        if (rightBucketBytes[bi] + rowSize > bucketBudget && rightBuckets[bi].length > 0) {
          await flushBucket(bi);
        }
        rightBuckets[bi].push(row);
        rightBucketBytes[bi] += rowSize;
      }
    }

    for (let bi = 0; bi < this.partitionCount; bi++) await flushBucket(bi);

    // Consume and partition left side with same bounded approach
    const leftBuckets: Row[][] = Array.from({ length: this.partitionCount }, () => []);
    const leftBucketBytes: number[] = new Array(this.partitionCount).fill(0);
    this.leftPartitionIds = new Array(this.partitionCount).fill("");

    const flushLeftBucket = async (bi: number): Promise<void> => {
      if (leftBuckets[bi].length === 0) return;
      const spillId = await this.spill!.writeRun(leftBuckets[bi]);
      this.leftPartitionIds[bi] = this.leftPartitionIds[bi]
        ? this.leftPartitionIds[bi] + "|" + spillId
        : spillId;
      leftBuckets[bi] = [];
      leftBucketBytes[bi] = 0;
    };

    while (true) {
      const batch = await this.left.next();
      if (!batch) break;
      for (const row of batch) {
        const bi = this.hashPartition(row[this.leftKey], this.partitionCount);
        const rowSize = estimateRowSize(row);
        if (leftBucketBytes[bi] + rowSize > bucketBudget && leftBuckets[bi].length > 0) {
          await flushLeftBucket(bi);
        }
        leftBuckets[bi].push(row);
        leftBucketBytes[bi] += rowSize;
      }
    }

    // Flush remaining left buckets
    for (let bi = 0; bi < this.partitionCount; bi++) await flushLeftBucket(bi);
  }

  /** Stream rows from a potentially multi-spill partition ("|"-delimited spill IDs). */
  private async *streamPartition(partitionIds: string): AsyncGenerator<Row> {
    if (!partitionIds) return;
    for (const spillId of partitionIds.split("|")) {
      yield* this.spill!.streamRun(spillId);
    }
  }

  /** Process one partition: load right → build map → probe left → emit matches. */
  private async processPartition(partIdx: number): Promise<Row[]> {
    // Load right partition into hash map
    const rightMap = new Map<string, Row[]>();
    for await (const row of this.streamPartition(this.rightPartitionIds[partIdx])) {
      const key = this.toJoinKey(row[this.rightKey]);
      const bucket = rightMap.get(key);
      if (bucket) bucket.push(row);
      else rightMap.set(key, [row]);
    }

    // Probe with left partition
    const result: Row[] = [];
    for await (const leftRow of this.streamPartition(this.leftPartitionIds[partIdx])) {
      const key = this.toJoinKey(leftRow[this.leftKey]);
      const rightRows = rightMap.get(key);
      if (rightRows) {
        for (const rightRow of rightRows) {
          result.push(this.mergeRow(leftRow, rightRow));
        }
      } else if (this.joinType === "left") {
        result.push({ ...leftRow });
      }
    }

    return result;
  }

  async next(): Promise<RowBatch | null> {
    await this.buildOrPartition();

    // In-memory path (no spill)
    if (this.hashMap) {
      while (true) {
        const batch = await this.left.next();
        if (!batch) return null;

        const result: Row[] = [];
        for (const leftRow of batch) {
          const key = this.toJoinKey(leftRow[this.leftKey]);
          const rightRows = this.hashMap.get(key);
          if (rightRows) {
            for (const rightRow of rightRows) {
              result.push(this.mergeRow(leftRow, rightRow));
            }
          } else if (this.joinType === "left") {
            result.push({ ...leftRow });
          }
        }

        if (result.length > 0) return result;
      }
    }

    // Partitioned spill path: process one partition at a time
    if (this.partitionedDrained) return null;

    while (this.currentPartition < this.partitionCount) {
      const rows = await this.processPartition(this.currentPartition);
      this.currentPartition++;
      if (rows.length > 0) return rows;
    }

    this.partitionedDrained = true;
    return null;
  }

  async close(): Promise<void> {
    this.hashMap = null;
    await this.left.close();
    await this.right.close();
  }
}

// ---------------------------------------------------------------------------
// Spill backends — filesystem (Node/Bun) and pluggable (R2 via SpillBackend)
// ---------------------------------------------------------------------------

/** Default memory budget for external sort: 256MB */
export const DEFAULT_MEMORY_BUDGET = 256 * 1024 * 1024;

/** Rough estimate of a row's memory footprint in bytes. */
export function estimateRowSize(row: Row): number {
  let size = 64; // object overhead
  for (const key in row) {
    const val = row[key];
    if (typeof val === "string") size += 40 + val.length * 2;
    else if (val instanceof Float32Array) size += 40 + val.byteLength;
    else size += 16;
  }
  return size;
}

// Re-export SpillBackend + columnar codec from r2-spill
export type { SpillBackend } from "./r2-spill.js";
export { encodeColumnarRun, decodeColumnarRun } from "./r2-spill.js";
import { encodeColumnarRun, decodeColumnarRun } from "./r2-spill.js";

/** Filesystem-backed spill for Node/Bun environments using columnar binary format. */
export class FsSpillBackend {
  private runFiles: string[] = [];
  private tmpDir: string | null = null;
  bytesWritten = 0;
  bytesRead = 0;

  async writeRun(rows: Row[]): Promise<string> {
    if (!this.tmpDir) {
      const fs = await import("node:fs/promises");
      const os = await import("node:os");
      const path = await import("node:path");
      this.tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "qm-sort-"));
    }
    const path = await import("node:path");
    const fs = await import("node:fs/promises");
    const runPath = path.join(this.tmpDir!, `run_${this.runFiles.length}.bin`);
    const buf = encodeColumnarRun(rows);
    this.bytesWritten += buf.byteLength;
    await fs.writeFile(runPath, new Uint8Array(buf));
    this.runFiles.push(runPath);
    return runPath;
  }

  async *streamRun(spillId: string): AsyncGenerator<Row> {
    const fs = await import("node:fs/promises");
    const fileData = await fs.readFile(spillId);
    const buf = fileData.buffer.slice(fileData.byteOffset, fileData.byteOffset + fileData.byteLength);
    this.bytesRead += buf.byteLength;
    yield* decodeColumnarRun(buf);
  }

  async cleanup(): Promise<void> {
    const fs = await import("node:fs/promises");
    for (const f of this.runFiles) await fs.unlink(f).catch(() => {});
    if (this.tmpDir) await fs.rmdir(this.tmpDir).catch(() => {});
    this.runFiles = [];
    this.tmpDir = null;
  }
}

// ---------------------------------------------------------------------------
// ExternalSortOperator — ORDER BY without LIMIT, spills via SpillBackend
// ---------------------------------------------------------------------------

const MERGE_BATCH_SIZE = 4096;

export class ExternalSortOperator implements Operator {
  private upstream: Operator;
  private col: string;
  private desc: boolean;
  private offset: number;
  private memoryBudget: number;
  private spill: FsSpillBackend | import("./r2-spill.js").SpillBackend;
  private runsGenerated = false;
  private spillIds: string[] = [];
  private inMemoryResult: Row[] | null = null;
  private inMemoryYielded = false;

  // K-way merge state (persists across next() calls)
  private mergeReaders: { iter: AsyncGenerator<Row>; current: Row | null }[] | null = null;
  private mergeHeap: number[] | null = null;
  private mergeCompareFn: ((a: Row, b: Row) => number) | null = null;
  private skipped = 0;

  constructor(
    upstream: Operator,
    sortColumn: string,
    desc: boolean,
    offset = 0,
    memoryBudget = DEFAULT_MEMORY_BUDGET,
    spill?: import("./r2-spill.js").SpillBackend,
  ) {
    this.upstream = upstream;
    this.col = sortColumn;
    this.desc = desc;
    this.offset = offset;
    this.memoryBudget = memoryBudget;
    this.spill = spill ?? new FsSpillBackend();
  }

  private getCompareFn(): (a: Row, b: Row) => number {
    if (this.mergeCompareFn) return this.mergeCompareFn;
    const col = this.col;
    const dir = this.desc ? -1 : 1;
    this.mergeCompareFn = (a: Row, b: Row): number => {
      const av = a[col], bv = b[col];
      if (av === null && bv === null) return 0;
      if (av === null) return 1;
      if (bv === null) return -1;
      return av < bv ? -dir : av > bv ? dir : 0;
    };
    return this.mergeCompareFn;
  }

  /** Phase 1: consume upstream, generate sorted runs. */
  private async generateRuns(): Promise<void> {
    if (this.runsGenerated) return;
    this.runsGenerated = true;

    const compareFn = this.getCompareFn();
    let currentRun: Row[] = [];
    let currentRunBytes = 0;

    const flushRun = async (): Promise<void> => {
      if (currentRun.length === 0) return;
      currentRun.sort(compareFn);
      const spillId = await this.spill.writeRun(currentRun);
      this.spillIds.push(spillId);
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

    // If everything fit in one run, keep in memory — no spill needed
    if (this.spillIds.length === 0) {
      currentRun.sort(compareFn);
      this.inMemoryResult = this.offset > 0 ? currentRun.slice(this.offset) : currentRun;
      return;
    }

    // Flush final run
    await flushRun();
  }

  /** Initialize k-way merge state. */
  private async initMerge(): Promise<void> {
    type RunReader = { iter: AsyncGenerator<Row>; current: Row | null };
    const readers: RunReader[] = [];

    for (const spillId of this.spillIds) {
      const iter = this.spill.streamRun(spillId);
      const first = await iter.next();
      const current = first.done ? null : first.value;
      readers.push({ iter, current });
    }

    const heap: number[] = [];
    for (let i = 0; i < readers.length; i++) {
      if (readers[i].current !== null) heap.push(i);
    }

    const compareFn = this.getCompareFn();
    const heapCmp = (a: number, b: number): number =>
      compareFn(readers[a].current!, readers[b].current!);

    const siftDown = (arr: number[], i: number): void => {
      while (true) {
        let t = i;
        const l = 2 * i + 1, r = 2 * i + 2;
        if (l < arr.length && heapCmp(arr[l], arr[t]) < 0) t = l;
        if (r < arr.length && heapCmp(arr[r], arr[t]) < 0) t = r;
        if (t === i) break;
        [arr[i], arr[t]] = [arr[t], arr[i]];
        i = t;
      }
    };

    for (let i = Math.floor(heap.length / 2) - 1; i >= 0; i--) siftDown(heap, i);

    this.mergeReaders = readers;
    this.mergeHeap = heap;
  }

  async next(): Promise<RowBatch | null> {
    await this.generateRuns();

    // In-memory path (no spill)
    if (this.inMemoryResult !== null) {
      if (this.inMemoryYielded) return null;
      this.inMemoryYielded = true;
      return this.inMemoryResult;
    }

    // Initialize merge on first call
    if (!this.mergeReaders) await this.initMerge();

    const readers = this.mergeReaders!;
    const heap = this.mergeHeap!;
    const compareFn = this.getCompareFn();

    const heapCmp = (a: number, b: number): number =>
      compareFn(readers[a].current!, readers[b].current!);

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

    if (heap.length === 0) {
      await this.spill.cleanup();
      return null;
    }

    // Yield one batch of MERGE_BATCH_SIZE rows from the merge
    const batch: Row[] = [];
    while (heap.length > 0 && batch.length < MERGE_BATCH_SIZE) {
      const minIdx = heap[0];
      const row = readers[minIdx].current!;

      // Handle offset — skip rows
      if (this.skipped < this.offset) {
        this.skipped++;
      } else {
        batch.push(row);
      }

      const next = await readers[minIdx].iter.next();
      if (next.done) {
        readers[minIdx].current = null;
        heap[0] = heap[heap.length - 1];
        heap.pop();
        if (heap.length > 0) siftDown(0);
      } else {
        readers[minIdx].current = next.value;
        siftDown(0);
      }
    }

    if (batch.length === 0) {
      // All rows were skipped by offset but merge exhausted
      await this.spill.cleanup();
      return null;
    }

    // If merge is now exhausted, cleanup
    if (heap.length === 0) {
      await this.spill.cleanup();
    }

    return batch;
  }

  async close(): Promise<void> {
    await this.spill.cleanup();
    await this.upstream.close();
  }
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
  /** Spill backend for external sort/join (default: FsSpillBackend). */
  spill?: import("./r2-spill.js").SpillBackend;
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
        query.offset ?? 0, memBudget, options?.spill,
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
 * Build an operator pipeline from a pre-existing scan operator (edge path).
 * Same logic as buildPipeline but doesn't create the ScanOperator itself.
 */
export function buildEdgePipeline(
  scan: Operator,
  query: QueryDescriptor,
  outputColumns: string[],
  options?: PipelineOptions,
): Operator {
  let pipeline: Operator = scan;
  const memBudget = options?.memoryBudgetBytes ?? DEFAULT_MEMORY_BUDGET;

  if (query.filters.length > 0) {
    pipeline = new FilterOperator(pipeline, query.filters);
  }

  const hasAgg = query.aggregates && query.aggregates.length > 0;

  if (hasAgg) {
    pipeline = new AggregateOperator(pipeline, query);
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
    if (query.limit !== undefined) {
      pipeline = new TopKOperator(
        pipeline, query.sortColumn, query.sortDirection === "desc",
        query.limit, query.offset ?? 0,
      );
    } else {
      pipeline = new ExternalSortOperator(
        pipeline, query.sortColumn, query.sortDirection === "desc",
        query.offset ?? 0, memBudget, options?.spill,
      );
    }
  } else if (query.offset || query.limit !== undefined) {
    pipeline = new LimitOperator(pipeline, query.limit ?? Infinity, query.offset ?? 0);
  }

  if (outputColumns.length > 0) {
    pipeline = new ProjectOperator(pipeline, outputColumns);
  }

  return pipeline;
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
