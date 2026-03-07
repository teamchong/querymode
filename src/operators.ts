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
  scanMs = 0;
  /** Whether this ScanOperator applies filters internally (WASM SIMD + JS fallback). */
  filtersApplied = false;
  /** Prefetched page data — fetched while decoding the previous page. */
  private prefetch: Promise<Map<string, { buf: ArrayBuffer; pageInfo: PageInfo }>> | null = null;
  private prefetchPageIdx = -1;
  private prefetchFragIdx = -1;
  private prefetchSkipped = 0;

  constructor(fragments: FragmentSource[], query: QueryDescriptor, wasm: WasmEngine, applyFilters = false) {
    this.fragments = fragments;
    this.query = query;
    this.wasm = wasm;
    this.filtersApplied = applyFilters && query.filters.length > 0;
  }

  /** Fetch specified columns for a page in parallel. Returns the pageInfoMap. */
  private fetchColumns(
    frag: FragmentSource, pi: number, cols: ColumnMeta[],
  ): { promise: Promise<Map<string, { buf: ArrayBuffer; pageInfo: PageInfo }>>; skipped: number } {
    const pageInfoMap = new Map<string, { buf: ArrayBuffer; pageInfo: PageInfo }>();
    const readPromises: Promise<void>[] = [];
    let skipped = 0;
    for (const col of cols) {
      const colPage = col.pages[pi];
      if (!colPage) continue;
      if (!this.query.vectorSearch && canSkipPage(colPage, this.query.filters, col.name)) {
        skipped++;
        continue;
      }
      readPromises.push(
        frag.readPage(col, colPage).then(buf => {
          this.bytesRead += buf.byteLength;
          pageInfoMap.set(col.name, { buf, pageInfo: colPage });
        }),
      );
    }
    return {
      promise: readPromises.length > 0
        ? Promise.all(readPromises).then(() => pageInfoMap)
        : Promise.resolve(pageInfoMap),
      skipped,
    };
  }

  /** Fetch all columns for a page. */
  private fetchPage(
    frag: FragmentSource, pi: number,
  ): { promise: Promise<Map<string, { buf: ArrayBuffer; pageInfo: PageInfo }>>; skipped: number } {
    return this.fetchColumns(frag, pi, frag.columns);
  }

  /** Find the next non-skippable page index starting from `start`. Returns -1 if none. */
  private findNextPage(frag: FragmentSource, start: number): number {
    const firstCol = frag.columns[0];
    if (!firstCol) return -1;
    for (let i = start; i < firstCol.pages.length; i++) {
      const page = firstCol.pages[i];
      if (this.query.vectorSearch || !canSkipPage(page, this.query.filters, firstCol.name)) {
        return i;
      }
    }
    return -1;
  }

  /** Start prefetching the next non-skippable page if available. */
  private startPrefetch(): void {
    // Find the next page to prefetch
    let fi = this.fragIdx;
    let pi = this.pageIdx;
    while (fi < this.fragments.length) {
      const frag = this.fragments[fi];
      const nextPi = this.findNextPage(frag, pi);
      if (nextPi >= 0) {
        const { promise, skipped } = this.fetchPage(frag, nextPi);
        this.prefetch = promise;
        this.prefetchPageIdx = nextPi;
        this.prefetchFragIdx = fi;
        this.prefetchSkipped = skipped;
        return;
      }
      fi++;
      pi = 0;
    }
    this.prefetch = null;
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

        const scanStart = Date.now();

        // Use prefetched data if available for this page, otherwise fetch now
        let pageInfoMap: Map<string, { buf: ArrayBuffer; pageInfo: PageInfo }>;
        if (this.prefetch && this.prefetchPageIdx === pi && this.prefetchFragIdx === this.fragIdx) {
          pageInfoMap = await this.prefetch;
          this.pagesSkipped += this.prefetchSkipped;
          this.prefetch = null;
        } else {
          const { promise, skipped } = this.fetchPage(frag, pi);
          this.pagesSkipped += skipped;
          pageInfoMap = await promise;
        }

        // Start prefetching the next page while we decode this one
        this.startPrefetch();

        // Late materialization: decode filter columns first, skip rest if no matches
        if (this.filtersApplied && this.query.filters.length > 0) {
          const filterColNames = new Set(this.query.filters.map(f => f.column));
          const filterCols = frag.columns.filter(c => filterColNames.has(c.name));
          const projCols = frag.columns.filter(c => !filterColNames.has(c.name));

          // Decode only filter columns from already-fetched page data
          const filterDecoded = decodePageBatch(pageInfoMap, filterCols, this.wasm);
          const firstDecoded = filterDecoded.values().next().value;
          if (!firstDecoded || firstDecoded.length === 0) continue;
          const rowCount = firstDecoded.length;

          // Apply filters to get matching indices
          const matchingIndices = scanFilterIndices(
            filterDecoded, filterCols, this.query.filters, rowCount, this.wasm,
          );

          // Skip decoding projection columns if nothing matched
          if (matchingIndices.length === 0) {
            this.scanMs += Date.now() - scanStart;
            continue;
          }

          // Decode projection columns (only needed now that we know rows match)
          const projDecoded = projCols.length > 0
            ? decodePageBatch(pageInfoMap, projCols, this.wasm)
            : new Map<string, DecodedValue[]>();

          // Merge filter + projection decoded data
          const allDecoded = new Map([...filterDecoded, ...projDecoded]);

          // Assemble rows only for matching indices
          const rows: Row[] = [];
          for (const idx of matchingIndices) {
            const row: Row = {};
            for (const col of frag.columns) {
              const vals = allDecoded.get(col.name);
              row[col.name] = vals ? (vals[idx] as Row[string]) : null;
            }
            rows.push(row);
          }
          this.scanMs += Date.now() - scanStart;
          if (rows.length > 0) return rows;
          continue;
        }

        // No filters — decode all columns, assemble all rows
        const decoded = decodePageBatch(pageInfoMap, frag.columns, this.wasm);
        const firstDecoded = decoded.values().next().value;
        if (!firstDecoded || firstDecoded.length === 0) continue;
        const rowCount = firstDecoded.length;

        const rows: Row[] = [];
        for (let i = 0; i < rowCount; i++) {
          const row: Row = {};
          for (const col of frag.columns) {
            const vals = decoded.get(col.name);
            row[col.name] = vals ? (vals[i] as Row[string]) : null;
          }
          rows.push(row);
        }
        this.scanMs += Date.now() - scanStart;
        if (rows.length > 0) return rows;
        continue;
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
// Scan-time filter — applies WASM SIMD for numeric columns, JS for rest
// ---------------------------------------------------------------------------

/** Map filter op string to WASM op code: 0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge */
function filterOpToWasm(op: string): number {
  switch (op) {
    case "eq": return 0;
    case "neq": return 1;
    case "lt": return 2;
    case "lte": return 3;
    case "gt": return 4;
    case "gte": return 5;
    default: return -1;
  }
}

/**
 * Compute matching row indices by applying filters before row materialization.
 * Uses WASM SIMD for numeric scalar filters, JS for string/bool/IN filters.
 * Returns the set of row indices that pass ALL filters.
 */
function scanFilterIndices(
  decoded: Map<string, DecodedValue[]>,
  columns: ColumnMeta[],
  filters: QueryDescriptor["filters"],
  rowCount: number,
  wasm: WasmEngine,
): Uint32Array {
  const colTypes = new Map(columns.map(c => [c.name, c.dtype]));

  // Start with all indices
  let indices: Uint32Array | null = null;

  for (const filter of filters) {
    const dtype = colTypes.get(filter.column);
    const values = decoded.get(filter.column);
    if (!values || values.length === 0) {
      // Column not decoded (skipped) — no rows match
      return new Uint32Array(0);
    }

    const wasmOp = filter.op !== "in" ? filterOpToWasm(filter.op) : -1;

    // Try WASM SIMD path for numeric scalar filters
    if (wasmOp >= 0 && typeof filter.value === "number" &&
        (dtype === "float64" || dtype === "float32" || dtype === "int32")) {
      const filterResult = wasmFilterNumeric(
        values, dtype, wasmOp, filter.value, rowCount, wasm,
      );
      if (filterResult) {
        indices = indices ? wasmIntersect(indices, filterResult, wasm) : filterResult;
        continue;
      }
    }

    // JS fallback: evaluate filter on current index set
    const src = indices ?? Uint32Array.from({ length: rowCount }, (_, i) => i);
    const kept: number[] = [];
    for (const idx of src) {
      const v = values[idx];
      if (v !== null && v !== undefined && matchesFilter(v as Row[string], filter)) {
        kept.push(idx);
      }
    }
    indices = new Uint32Array(kept);
  }

  return indices ?? Uint32Array.from({ length: rowCount }, (_, i) => i);
}

/** Run WASM filterFloat64Buffer or filterInt32Buffer on decoded numeric values. */
function wasmFilterNumeric(
  values: DecodedValue[],
  dtype: string,
  op: number,
  filterValue: number,
  rowCount: number,
  wasm: WasmEngine,
): Uint32Array | null {
  try {
    wasm.exports.resetHeap();

    if (dtype === "float64" || dtype === "float32") {
      // Pack values into Float64Array
      const dataPtr = wasm.exports.alloc(rowCount * 8);
      if (!dataPtr) return null;
      const dst = new Float64Array(wasm.exports.memory.buffer, dataPtr, rowCount);
      for (let i = 0; i < rowCount; i++) {
        dst[i] = (values[i] as number) ?? 0;
      }
      // Allocate output indices
      const outPtr = wasm.exports.alloc(rowCount * 4);
      if (!outPtr) return null;
      const count = wasm.exports.filterFloat64Buffer(
        dataPtr, rowCount, op, filterValue, outPtr, rowCount,
      );
      return new Uint32Array(wasm.exports.memory.buffer.slice(outPtr, outPtr + count * 4));
    }

    if (dtype === "int32") {
      const dataPtr = wasm.exports.alloc(rowCount * 4);
      if (!dataPtr) return null;
      const dst = new Int32Array(wasm.exports.memory.buffer, dataPtr, rowCount);
      for (let i = 0; i < rowCount; i++) {
        dst[i] = (values[i] as number) ?? 0;
      }
      const outPtr = wasm.exports.alloc(rowCount * 4);
      if (!outPtr) return null;
      const count = wasm.exports.filterInt32Buffer(
        dataPtr, rowCount, op, filterValue, outPtr, rowCount,
      );
      return new Uint32Array(wasm.exports.memory.buffer.slice(outPtr, outPtr + count * 4));
    }

    return null;
  } catch {
    return null; // WASM failure — fall through to JS
  }
}

/** Intersect two sorted index arrays using WASM. */
function wasmIntersect(a: Uint32Array, b: Uint32Array, wasm: WasmEngine): Uint32Array {
  try {
    wasm.exports.resetHeap();
    const aPtr = wasm.exports.alloc(a.byteLength);
    const bPtr = wasm.exports.alloc(b.byteLength);
    const outPtr = wasm.exports.alloc(Math.min(a.length, b.length) * 4);
    if (!aPtr || !bPtr || !outPtr) {
      // Fallback: JS intersect
      const setB = new Set(b);
      return a.filter(v => setB.has(v));
    }
    new Uint32Array(wasm.exports.memory.buffer, aPtr, a.length).set(a);
    new Uint32Array(wasm.exports.memory.buffer, bPtr, b.length).set(b);
    const count = wasm.exports.intersectIndices(
      aPtr, a.length, bPtr, b.length, outPtr, Math.min(a.length, b.length),
    );
    return new Uint32Array(wasm.exports.memory.buffer.slice(outPtr, outPtr + count * 4));
  } catch {
    const setB = new Set(b);
    return a.filter(v => setB.has(v));
  }
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
// ComputedColumnOperator — adds computed columns via callbacks
// ---------------------------------------------------------------------------

export class ComputedColumnOperator implements Operator {
  private upstream: Operator;
  private computations: { alias: string; fn: (row: Row) => unknown }[];

  constructor(upstream: Operator, computations: { alias: string; fn: (row: Row) => unknown }[]) {
    this.upstream = upstream;
    this.computations = computations;
  }

  async next(): Promise<RowBatch | null> {
    const batch = await this.upstream.next();
    if (!batch) return null;

    for (const row of batch) {
      for (const comp of this.computations) {
        row[comp.alias] = comp.fn(row) as Row[string];
      }
    }
    return batch;
  }

  async close(): Promise<void> {
    await this.upstream.close();
  }
}

// ---------------------------------------------------------------------------
// SubqueryInOperator — filters rows where column value is in a pre-computed set
// ---------------------------------------------------------------------------

export class SubqueryInOperator implements Operator {
  private upstream: Operator;
  private column: string;
  private valueSet: Set<string>;

  constructor(upstream: Operator, column: string, valueSet: Set<string>) {
    this.upstream = upstream;
    this.column = column;
    this.valueSet = valueSet;
  }

  async next(): Promise<RowBatch | null> {
    while (true) {
      const batch = await this.upstream.next();
      if (!batch) return null;

      const filtered: Row[] = [];
      for (const row of batch) {
        const val = row[this.column];
        const key = val === null ? "__null__" : typeof val === "bigint" ? val.toString() : String(val);
        if (this.valueSet.has(key)) {
          filtered.push(row);
        }
      }
      if (filtered.length > 0) return filtered;
    }
  }

  async close(): Promise<void> {
    await this.upstream.close();
  }
}

// ---------------------------------------------------------------------------
// WindowOperator — window functions with partition-and-sort
// ---------------------------------------------------------------------------

import type { WindowSpec } from "./types.js";

export class WindowOperator implements Operator {
  private upstream: Operator;
  private windows: WindowSpec[];
  private consumed = false;

  constructor(upstream: Operator, windows: WindowSpec[]) {
    this.upstream = upstream;
    this.windows = windows;
  }

  async next(): Promise<RowBatch | null> {
    if (this.consumed) return null;
    this.consumed = true;

    // Collect all rows (window functions need full partition)
    const allRows: Row[] = [];
    while (true) {
      const batch = await this.upstream.next();
      if (!batch) break;
      for (const row of batch) allRows.push(row);
    }

    if (allRows.length === 0) return null;

    for (const win of this.windows) {
      this.evaluateWindow(allRows, win);
    }

    return allRows;
  }

  private evaluateWindow(rows: Row[], win: WindowSpec): void {
    // Group by partitionBy keys
    const partitions = new Map<string, number[]>();
    for (let i = 0; i < rows.length; i++) {
      const key = win.partitionBy.length > 0
        ? win.partitionBy.map(c => String(rows[i][c] ?? "")).join("\x00")
        : "__all__";
      const indices = partitions.get(key);
      if (indices) indices.push(i);
      else partitions.set(key, [i]);
    }

    // Process each partition
    for (const indices of partitions.values()) {
      // Sort indices by orderBy
      if (win.orderBy.length > 0) {
        indices.sort((a, b) => {
          for (const ob of win.orderBy) {
            const av = rows[a][ob.column], bv = rows[b][ob.column];
            if (av === null && bv === null) continue;
            if (av === null) return ob.direction === "asc" ? 1 : -1;
            if (bv === null) return ob.direction === "asc" ? -1 : 1;
            if (av < bv) return ob.direction === "asc" ? -1 : 1;
            if (av > bv) return ob.direction === "asc" ? 1 : -1;
          }
          return 0;
        });
      }

      this.applyWindowFn(rows, indices, win);
    }
  }

  private applyWindowFn(rows: Row[], indices: number[], win: WindowSpec): void {
    const { fn, alias, orderBy } = win;

    switch (fn) {
      case "row_number":
        for (let i = 0; i < indices.length; i++) {
          rows[indices[i]][alias] = i + 1;
        }
        break;

      case "rank": {
        let rank = 1;
        for (let i = 0; i < indices.length; i++) {
          if (i > 0 && !this.orderByEqual(rows, indices[i], indices[i - 1], orderBy)) {
            rank = i + 1;
          }
          rows[indices[i]][alias] = rank;
        }
        break;
      }

      case "dense_rank": {
        let rank = 1;
        for (let i = 0; i < indices.length; i++) {
          if (i > 0 && !this.orderByEqual(rows, indices[i], indices[i - 1], orderBy)) {
            rank++;
          }
          rows[indices[i]][alias] = rank;
        }
        break;
      }

      case "lag": {
        const offset = win.args?.offset ?? 1;
        const defaultVal = win.args?.default_ ?? null;
        for (let i = 0; i < indices.length; i++) {
          const srcIdx = i - offset;
          if (srcIdx >= 0 && srcIdx < indices.length && orderBy.length > 0) {
            rows[indices[i]][alias] = rows[indices[srcIdx]][orderBy[0].column] as Row[string];
          } else {
            rows[indices[i]][alias] = defaultVal as Row[string];
          }
        }
        break;
      }

      case "lead": {
        const offset = win.args?.offset ?? 1;
        const defaultVal = win.args?.default_ ?? null;
        for (let i = 0; i < indices.length; i++) {
          const srcIdx = i + offset;
          if (srcIdx >= 0 && srcIdx < indices.length && orderBy.length > 0) {
            rows[indices[i]][alias] = rows[indices[srcIdx]][orderBy[0].column] as Row[string];
          } else {
            rows[indices[i]][alias] = defaultVal as Row[string];
          }
        }
        break;
      }

      case "sum": case "avg": case "min": case "max": case "count": {
        const col = orderBy.length > 0 ? orderBy[0].column : "";
        this.applyAggregateWindow(rows, indices, win, col);
        break;
      }
    }
  }

  private applyAggregateWindow(rows: Row[], indices: number[], win: WindowSpec, col: string): void {
    const { fn, alias, frame } = win;
    const frameType = frame?.type ?? "range";
    const frameStart = frame?.start ?? "unbounded";
    const frameEnd = frame?.end ?? "current";

    for (let i = 0; i < indices.length; i++) {
      let start: number, end: number;

      if (frameStart === "unbounded") start = 0;
      else start = Math.max(0, i + (frameStart as number));

      if (frameEnd === "unbounded") end = indices.length - 1;
      else if (frameEnd === "current") end = i;
      else end = Math.min(indices.length - 1, i + (frameEnd as number));

      let sum = 0, count = 0, min = Infinity, max = -Infinity;
      for (let j = start; j <= end; j++) {
        const val = rows[indices[j]][col];
        if (val === null || val === undefined) continue;
        const n = typeof val === "number" ? val : typeof val === "bigint" ? Number(val) : 0;
        sum += n;
        count++;
        if (n < min) min = n;
        if (n > max) max = n;
      }

      switch (fn) {
        case "sum": rows[indices[i]][alias] = sum; break;
        case "avg": rows[indices[i]][alias] = count === 0 ? 0 : sum / count; break;
        case "min": rows[indices[i]][alias] = min === Infinity ? null : min; break;
        case "max": rows[indices[i]][alias] = max === -Infinity ? null : max; break;
        case "count": rows[indices[i]][alias] = count; break;
      }
    }
  }

  private orderByEqual(rows: Row[], idx1: number, idx2: number, orderBy: WindowSpec["orderBy"]): boolean {
    for (const ob of orderBy) {
      const a = rows[idx1][ob.column], b = rows[idx2][ob.column];
      if (a !== b) return false;
    }
    return true;
  }

  async close(): Promise<void> {
    await this.upstream.close();
  }
}

// ---------------------------------------------------------------------------
// DistinctOperator — deduplication by column keys
// ---------------------------------------------------------------------------

export class DistinctOperator implements Operator {
  private upstream: Operator;
  private columns: string[];
  private seen = new Set<string>();

  constructor(upstream: Operator, columns: string[]) {
    this.upstream = upstream;
    this.columns = columns;
  }

  async next(): Promise<RowBatch | null> {
    while (true) {
      const batch = await this.upstream.next();
      if (!batch) return null;

      const unique: Row[] = [];
      for (const row of batch) {
        const key = this.columns.length > 0
          ? this.columns.map(c => String(row[c] ?? "")).join("\x00")
          : Object.keys(row).map(k => String(row[k] ?? "")).join("\x00");
        if (!this.seen.has(key)) {
          this.seen.add(key);
          unique.push(row);
        }
      }
      if (unique.length > 0) return unique;
    }
  }

  async close(): Promise<void> {
    this.seen.clear();
    await this.upstream.close();
  }
}

// ---------------------------------------------------------------------------
// SetOperator — union/intersect/except of two operator streams
// ---------------------------------------------------------------------------

export class SetOperator implements Operator {
  private left: Operator;
  private right: Operator;
  private mode: "union" | "union_all" | "intersect" | "except";
  private phase: "left" | "right" | "done" = "left";
  private seen: Set<string> | null = null;
  private rightKeys: Set<string> | null = null;

  constructor(
    left: Operator,
    right: Operator,
    mode: "union" | "union_all" | "intersect" | "except",
  ) {
    this.left = left;
    this.right = right;
    this.mode = mode;
    if (mode !== "union_all") this.seen = new Set();
  }

  private rowKey(row: Row): string {
    return Object.keys(row).sort().map(k => `${k}=${String(row[k] ?? "")}`).join("\x00");
  }

  async next(): Promise<RowBatch | null> {
    if (this.mode === "union_all") {
      return this.nextUnionAll();
    }
    if (this.mode === "union") {
      return this.nextUnion();
    }
    if (this.mode === "intersect") {
      return this.nextIntersect();
    }
    if (this.mode === "except") {
      return this.nextExcept();
    }
    return null;
  }

  private async nextUnionAll(): Promise<RowBatch | null> {
    if (this.phase === "left") {
      const batch = await this.left.next();
      if (batch) return batch;
      this.phase = "right";
    }
    if (this.phase === "right") {
      const batch = await this.right.next();
      if (batch) return batch;
      this.phase = "done";
    }
    return null;
  }

  private async nextUnion(): Promise<RowBatch | null> {
    while (this.phase !== "done") {
      const source = this.phase === "left" ? this.left : this.right;
      const batch = await source.next();
      if (!batch) {
        if (this.phase === "left") { this.phase = "right"; continue; }
        this.phase = "done";
        return null;
      }
      const unique: Row[] = [];
      for (const row of batch) {
        const key = this.rowKey(row);
        if (!this.seen!.has(key)) {
          this.seen!.add(key);
          unique.push(row);
        }
      }
      if (unique.length > 0) return unique;
    }
    return null;
  }

  private async nextIntersect(): Promise<RowBatch | null> {
    // First call: collect all right keys
    if (!this.rightKeys) {
      this.rightKeys = new Set();
      while (true) {
        const batch = await this.right.next();
        if (!batch) break;
        for (const row of batch) this.rightKeys.add(this.rowKey(row));
      }
    }

    while (true) {
      const batch = await this.left.next();
      if (!batch) return null;
      const matches: Row[] = [];
      for (const row of batch) {
        const key = this.rowKey(row);
        if (this.rightKeys.has(key) && !this.seen!.has(key)) {
          this.seen!.add(key);
          matches.push(row);
        }
      }
      if (matches.length > 0) return matches;
    }
  }

  private async nextExcept(): Promise<RowBatch | null> {
    // First call: collect all right keys
    if (!this.rightKeys) {
      this.rightKeys = new Set();
      while (true) {
        const batch = await this.right.next();
        if (!batch) break;
        for (const row of batch) this.rightKeys.add(this.rowKey(row));
      }
    }

    while (true) {
      const batch = await this.left.next();
      if (!batch) return null;
      const unmatched: Row[] = [];
      for (const row of batch) {
        const key = this.rowKey(row);
        if (!this.rightKeys.has(key) && !this.seen!.has(key)) {
          this.seen!.add(key);
          unmatched.push(row);
        }
      }
      if (unmatched.length > 0) return unmatched;
    }
  }

  async close(): Promise<void> {
    this.seen?.clear();
    this.rightKeys?.clear();
    await this.left.close();
    await this.right.close();
  }
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
  scanMs = 0;

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

    const scanStart = Date.now();
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
            if (agg.fn === "sum" || agg.fn === "avg") {
              acc[ai].sum += Number(this.wasm.sumInt64(buf));
            }
            if (agg.fn === "min") {
              const v = Number(this.wasm.minInt64(buf));
              if (v < acc[ai].min) acc[ai].min = v;
            }
            if (agg.fn === "max") {
              const v = Number(this.wasm.maxInt64(buf));
              if (v > acc[ai].max) acc[ai].max = v;
            }
          }
        }
      }
    }
    this.scanMs += Date.now() - scanStart;

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
  private joinType: "inner" | "left" | "right" | "full" | "cross";
  private memoryBudget: number;
  private spill: import("./r2-spill.js").SpillBackend | null;

  // In-memory build phase state
  private hashMap: Map<string, Row[]> | null = null;
  private buildExceeded = false;
  private buildSizeBytes = 0;

  // For right/full joins: track which right rows were matched
  private rightMatched: Set<string> | null = null;
  private rightUnmatchedEmitted = false;

  // For cross join: buffer right side, iterate left × right
  private crossRightBuffer: Row[] | null = null;
  private crossLeftRow: Row | null = null;
  private crossLeftBatch: Row[] | null = null;
  private crossLeftIdx = 0;
  private crossRightIdx = 0;
  private crossDone = false;

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
    joinType: "inner" | "left" | "right" | "full" | "cross" = "inner",
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

  /** Create a row ID for tracking matched right rows (for right/full joins). */
  private rightRowId(key: string, idx: number): string {
    return `${key}\x00${idx}`;
  }

  /** Emit unmatched right rows with null-filled left columns. */
  private emitUnmatchedRight(): Row[] {
    if (!this.hashMap || !this.rightMatched) return [];
    const result: Row[] = [];
    for (const [key, rows] of this.hashMap) {
      for (let i = 0; i < rows.length; i++) {
        if (!this.rightMatched.has(this.rightRowId(key, i))) {
          result.push({ ...rows[i] });
        }
      }
    }
    return result;
  }

  /** Try to build hash map in memory. If build side exceeds budget, switch to partition mode. */
  private async buildOrPartition(): Promise<void> {
    if (this.hashMap || this.partitionCount > 0 || this.crossRightBuffer) return;

    // Cross join: buffer entire right side (no key-based hash)
    if (this.joinType === "cross") {
      this.crossRightBuffer = [];
      while (true) {
        const batch = await this.right.next();
        if (!batch) break;
        this.crossRightBuffer.push(...batch);
      }
      return;
    }

    // Initialize match tracker for right/full joins
    if (this.joinType === "right" || this.joinType === "full") {
      this.rightMatched = new Set();
    }

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
      let exceededBatchTail: Row[] = [];

      while (true) {
        const batch = await this.right.next();
        if (!batch) break;
        for (let ri = 0; ri < batch.length; ri++) {
          const row = batch[ri];
          const rowSize = estimateRowSize(row);
          this.buildSizeBytes += rowSize;
          inMemoryRows.push(row);

          if (this.spill && this.buildSizeBytes > this.memoryBudget) {
            exceeds = true;
            // Capture remaining rows in this batch that we haven't visited
            exceededBatchTail = batch.slice(ri + 1);
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

      // Partition the remaining rows from the batch that triggered the exceed
      if (exceededBatchTail.length > 0) {
        await this.partitionRightRows(exceededBatchTail);
      }
      // Continue consuming remaining right-side batches
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

    // Track matched right rows for right/full joins
    const matched = (this.joinType === "right" || this.joinType === "full") ? new Set<string>() : null;

    // Probe with left partition
    const result: Row[] = [];
    for await (const leftRow of this.streamPartition(this.leftPartitionIds[partIdx])) {
      const key = this.toJoinKey(leftRow[this.leftKey]);
      const rightRows = rightMap.get(key);
      if (rightRows) {
        for (let i = 0; i < rightRows.length; i++) {
          result.push(this.mergeRow(leftRow, rightRows[i]));
          if (matched) matched.add(`${key}\x00${i}`);
        }
      } else if (this.joinType === "left" || this.joinType === "full") {
        result.push({ ...leftRow });
      }
    }

    // Emit unmatched right rows for right/full joins
    if (matched) {
      for (const [key, rows] of rightMap) {
        for (let i = 0; i < rows.length; i++) {
          if (!matched.has(`${key}\x00${i}`)) {
            result.push({ ...rows[i] });
          }
        }
      }
    }

    return result;
  }

  async next(): Promise<RowBatch | null> {
    await this.buildOrPartition();

    // Cross join path: nested loop (left × right)
    if (this.crossRightBuffer) {
      if (this.crossDone || this.crossRightBuffer.length === 0) return null;

      while (true) {
        // Need a new left batch?
        if (!this.crossLeftBatch || this.crossLeftIdx >= this.crossLeftBatch.length) {
          const batch = await this.left.next();
          if (!batch) { this.crossDone = true; return null; }
          this.crossLeftBatch = batch;
          this.crossLeftIdx = 0;
          this.crossRightIdx = 0;
        }

        const result: Row[] = [];
        const leftRow = this.crossLeftBatch[this.crossLeftIdx];
        const batchLimit = 1024;
        while (this.crossRightIdx < this.crossRightBuffer.length && result.length < batchLimit) {
          result.push(this.mergeRow(leftRow, this.crossRightBuffer[this.crossRightIdx]));
          this.crossRightIdx++;
        }

        if (this.crossRightIdx >= this.crossRightBuffer.length) {
          this.crossLeftIdx++;
          this.crossRightIdx = 0;
        }

        if (result.length > 0) return result;
      }
    }

    // In-memory path (no spill)
    if (this.hashMap) {
      while (true) {
        const batch = await this.left.next();
        if (!batch) {
          // After exhausting left side, emit unmatched right rows for right/full joins
          if (!this.rightUnmatchedEmitted && (this.joinType === "right" || this.joinType === "full")) {
            this.rightUnmatchedEmitted = true;
            const unmatched = this.emitUnmatchedRight();
            if (unmatched.length > 0) return unmatched;
          }
          return null;
        }

        const result: Row[] = [];
        for (const leftRow of batch) {
          const key = this.toJoinKey(leftRow[this.leftKey]);
          const rightRows = this.hashMap.get(key);
          if (rightRows) {
            for (let i = 0; i < rightRows.length; i++) {
              result.push(this.mergeRow(leftRow, rightRows[i]));
              if (this.rightMatched) this.rightMatched.add(this.rightRowId(key, i));
            }
          } else if (this.joinType === "left" || this.joinType === "full") {
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
    this.rightMatched = null;
    this.crossRightBuffer = null;
    this.crossLeftBatch = null;
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

  const scan = new ScanOperator(fragments, query, wasm, /* applyFilters */ true);
  let pipeline: Operator = scan;
  const memBudget = options?.memoryBudgetBytes ?? DEFAULT_MEMORY_BUDGET;

  // Filter — skip if ScanOperator already applies filters via WASM SIMD
  if (query.filters.length > 0 && !scan.filtersApplied) {
    pipeline = new FilterOperator(pipeline, query.filters);
  }

  // SubqueryIn filters (pre-computed value sets)
  if (query.subqueryIn) {
    for (const sq of query.subqueryIn) {
      pipeline = new SubqueryInOperator(pipeline, sq.column, sq.valueSet);
    }
  }

  // Computed columns (in-process callbacks)
  if (query.computedColumns) {
    pipeline = new ComputedColumnOperator(pipeline, query.computedColumns);
  }

  // Window functions
  if (query.windows && query.windows.length > 0) {
    pipeline = new WindowOperator(pipeline, query.windows);
  }

  // Distinct
  if (query.distinct) {
    pipeline = new DistinctOperator(pipeline, query.distinct);
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

  // SubqueryIn filters
  if (query.subqueryIn) {
    for (const sq of query.subqueryIn) {
      pipeline = new SubqueryInOperator(pipeline, sq.column, sq.valueSet);
    }
  }

  // Computed columns
  if (query.computedColumns) {
    pipeline = new ComputedColumnOperator(pipeline, query.computedColumns);
  }

  // User-injected pipe stages (inserted after filter/computed, before agg/sort)
  if (query.pipeStages) {
    for (const stage of query.pipeStages) {
      pipeline = stage(pipeline);
    }
  }

  // Window functions
  if (query.windows && query.windows.length > 0) {
    pipeline = new WindowOperator(pipeline, query.windows);
  }

  // Distinct
  if (query.distinct) {
    pipeline = new DistinctOperator(pipeline, query.distinct);
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

  // Skip projection when aggregation is active — aggregate output columns
  // (aliases like "count_*", "sum_value") don't match the original table columns,
  // so projecting to table columns would strip all aggregate results.
  if (!hasAgg && outputColumns.length > 0) {
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
