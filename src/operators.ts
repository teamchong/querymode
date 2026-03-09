/**
 * Streaming operator pipeline for QueryMode.
 *
 * Pull-based: the final operator pulls batches lazily from upstream.
 * Each operator processes one batch at a time — bounded memory.
 *
 * Pipeline: Scan → Filter → Aggregate|TopK|Sort|Limit → Project
 */

import type { ColumnMeta, FilterOp, PageInfo, Row } from "./types.js";
import { NULL_SENTINEL } from "./types.js";
import type { QueryDescriptor } from "./client.js";
import type { WasmEngine } from "./wasm-engine.js";
import { canSkipPage, matchesFilter, rowPassesFilters, decodePage } from "./decode.js";
import { decodeParquetColumnChunk } from "./parquet-decode.js";

const _textEncoder = new TextEncoder();

/** Cached identity index arrays to avoid repeated allocations on hot paths. */
const _identityCache = new Map<number, Uint32Array>();
export function identityIndices(n: number): Uint32Array {
  let cached = _identityCache.get(n);
  if (cached) return cached;
  cached = new Uint32Array(n);
  for (let i = 0; i < n; i++) cached[i] = i;
  // Keep cache bounded — only cache common page sizes
  if (_identityCache.size > 16) _identityCache.clear();
  _identityCache.set(n, cached);
  return cached;
}

export type DecodedValue = number | bigint | string | boolean | Float32Array | null;
import {
  computePartialAgg,
  computePartialAggColumnar,
  mergePartialAggs,
  finalizePartialAgg,
  type PartialAgg,
} from "./partial-agg.js";

/** A batch of rows flowing through the pipeline. */
export type RowBatch = Row[];

/** Columnar batch — column-oriented data flowing through the pipeline.
 *  Each column maps to an array of decoded values (typed arrays for numerics, string[] for strings).
 *  Selection vector (optional) identifies active row indices (post-filter).
 *  When selection is present, only those indices are valid; columns still hold the full page data. */
export interface ColumnarBatch {
  /** Column data keyed by name. */
  columns: Map<string, DecodedValue[]>;
  /** Number of logical rows (length of columns, not selection). */
  rowCount: number;
  /** Active row indices — if set, only these rows are "alive" (post-filter). */
  selection?: Uint32Array;
}

/** Pull-based operator interface. */
export interface Operator {
  /** Pull the next batch of rows. Returns null when exhausted. */
  next(): Promise<RowBatch | null>;
  /** Pull the next columnar batch. Returns null when exhausted.
   *  Operators that support columnar mode implement this; callers check before using. */
  nextColumnar?(): Promise<ColumnarBatch | null>;
  /** Release resources. */
  close(): Promise<void>;
}

/** Materialize a ColumnarBatch into Row[] — used at pipeline boundaries. */
export function materializeRows(batch: ColumnarBatch): Row[] {
  const rows: Row[] = [];
  const indices = batch.selection ?? identityIndices(batch.rowCount);
  for (const idx of indices) {
    const row: Row = {};
    for (const [name, vals] of batch.columns) {
      row[name] = vals[idx] as Row[string] ?? null;
    }
    rows.push(row);
  }
  return rows;
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

/** Check if a page can be skipped by checking ALL filter columns' min/max stats.
 *  Returns true if ANY AND filter proves zero matches, or if ALL OR groups prove zero matches. */
export function canSkipPageMultiCol(
  columns: ColumnMeta[], pageIdx: number, filters: QueryDescriptor["filters"],
  filterGroups?: FilterOp[][],
): boolean {
  const colMap = new Map(columns.map(c => [c.name, c]));

  // Check AND filters: skip if ANY single filter eliminates the page
  for (const f of filters) {
    const col = colMap.get(f.column);
    if (!col) continue;
    const page = col.pages[pageIdx];
    if (!page) continue;
    if (canSkipPage(page, [f], f.column)) return true;
  }

  // Check OR groups: skip only if ALL groups eliminate the page
  if (filterGroups && filterGroups.length > 0) {
    const allGroupsSkip = filterGroups.every(group =>
      canSkipAndGroup(colMap, pageIdx, group),
    );
    if (allGroupsSkip) return true;
  }

  return false;
}

/** Check if an AND-connected filter group can be proven to produce zero matches for a page. */
function canSkipAndGroup(
  colMap: Map<string, ColumnMeta>, pageIdx: number, filters: FilterOp[],
): boolean {
  for (const f of filters) {
    const col = colMap.get(f.column);
    if (!col) continue;
    const page = col.pages[pageIdx];
    if (!page) continue;
    if (canSkipPage(page, [f], f.column)) return true;
  }
  return false;
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
  private _filterColNames: Set<string> | null = null;

  constructor(fragments: FragmentSource[], query: QueryDescriptor, wasm: WasmEngine, applyFilters = false) {
    this.fragments = fragments;
    this.query = query;
    this.wasm = wasm;
    this.filtersApplied = applyFilters && (query.filters.length > 0 || !!(query.filterGroups && query.filterGroups.length > 0));
    if (this.filtersApplied) {
      this._filterColNames = new Set<string>();
      for (const f of query.filters) this._filterColNames.add(f.column);
      if (query.filterGroups) for (const g of query.filterGroups) for (const f of g) this._filterColNames.add(f.column);
    }
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
      if (this.query.vectorSearch || !canSkipPageMultiCol(frag.columns, i, this.query.filters, this.query.filterGroups)) {
        return i;
      }
    }
    return -1;
  }

  /** Start prefetching the next non-skippable page if available.
   *  When filters are applied, only prefetch filter columns (phase 1 of two-phase). */
  private startPrefetch(): void {
    let fi = this.fragIdx;
    let pi = this.pageIdx;
    while (fi < this.fragments.length) {
      const frag = this.fragments[fi];
      const nextPi = this.findNextPage(frag, pi);
      if (nextPi >= 0) {
        // Two-phase: only prefetch filter columns to minimize R2 I/O
        let cols: ColumnMeta[];
        if (this.filtersApplied && this._filterColNames) {
          cols = frag.columns.filter(c => this._filterColNames!.has(c.name));
        } else {
          cols = frag.columns;
        }
        const { promise, skipped } = this.fetchColumns(frag, nextPi, cols);
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

  async nextColumnar(): Promise<ColumnarBatch | null> {
    while (this.fragIdx < this.fragments.length) {
      const frag = this.fragments[this.fragIdx];
      const firstCol = frag.columns[0];
      if (!firstCol) { this.fragIdx++; this.pageIdx = 0; continue; }

      const totalPages = firstCol.pages.length;

      while (this.pageIdx < totalPages) {
        const pi = this.pageIdx;
        this.pageIdx++;

        // Page-level skip via min/max stats — check ALL filter columns, not just first
        if (!this.query.vectorSearch && canSkipPageMultiCol(frag.columns, pi, this.query.filters, this.query.filterGroups)) {
          this.pagesSkipped += frag.columns.length;
          continue;
        }

        const scanStart = Date.now();

        // Resolve prefetch or fetch fresh
        let pageInfoMap: Map<string, { buf: ArrayBuffer; pageInfo: PageInfo }>;
        const hasPrefetch = this.prefetch && this.prefetchPageIdx === pi && this.prefetchFragIdx === this.fragIdx;
        if (hasPrefetch) {
          pageInfoMap = await this.prefetch!;
          this.pagesSkipped += this.prefetchSkipped;
          this.prefetch = null;
        } else {
          pageInfoMap = new Map(); // will be populated in phase-specific fetch below
        }

        // Start prefetching the next page while we process this one
        this.startPrefetch();

        // Two-phase late materialization: fetch+decode filter cols first, skip projection I/O if no matches
        const allFilters = this.query.filters;
        const allFilterGroups = this.query.filterGroups;
        const hasFilters = allFilters.length > 0 || (allFilterGroups && allFilterGroups.length > 0);
        if (this.filtersApplied && hasFilters) {
          const fcn = this._filterColNames!;
          const filterCols = frag.columns.filter(c => fcn.has(c.name));
          const projCols = frag.columns.filter(c => !fcn.has(c.name));

          // Phase 1: Fetch + decode only filter columns
          let filterPageMap: Map<string, { buf: ArrayBuffer; pageInfo: PageInfo }>;
          if (hasPrefetch) {
            // Prefetch already fetched filter columns (two-phase prefetch)
            filterPageMap = pageInfoMap;
          } else {
            const { promise, skipped } = this.fetchColumns(frag, pi, filterCols);
            this.pagesSkipped += skipped;
            filterPageMap = await promise;
          }

          const filterDecoded = decodePageBatch(filterPageMap, filterCols, this.wasm);
          const firstDecoded = filterDecoded.values().next().value;
          if (!firstDecoded || firstDecoded.length === 0) {
            this.scanMs += Date.now() - scanStart;
            continue;
          }
          const rowCount = firstDecoded.length;

          const matchingIndices = scanFilterIndices(
            filterDecoded, filterCols, this.query.filters, rowCount, this.wasm,
            this.query.filterGroups,
          );

          if (matchingIndices.length === 0) {
            this.scanMs += Date.now() - scanStart;
            continue; // Skip projection column I/O entirely
          }

          // Phase 2: Fetch + decode projection columns (only needed because rows matched)
          let projDecoded: Map<string, DecodedValue[]>;
          if (projCols.length > 0) {
            const { promise: projPromise, skipped } = this.fetchColumns(frag, pi, projCols);
            this.pagesSkipped += skipped;
            const projPageMap = await projPromise;
            projDecoded = decodePageBatch(projPageMap, projCols, this.wasm);
          } else {
            projDecoded = new Map<string, DecodedValue[]>();
          }

          const allDecoded = new Map([...filterDecoded, ...projDecoded]);
          this.scanMs += Date.now() - scanStart;

          if (matchingIndices.length > 0) {
            return { columns: allDecoded, rowCount, selection: matchingIndices };
          }
          continue;
        }

        // No filters — fetch + decode all columns
        if (!hasPrefetch) {
          const { promise, skipped } = this.fetchPage(frag, pi);
          this.pagesSkipped += skipped;
          pageInfoMap = await promise;
        }
        const decoded = decodePageBatch(pageInfoMap, frag.columns, this.wasm);
        const firstDecoded = decoded.values().next().value;
        if (!firstDecoded || firstDecoded.length === 0) continue;
        const rowCount = firstDecoded.length;

        this.scanMs += Date.now() - scanStart;
        if (rowCount > 0) return { columns: decoded, rowCount };
        continue;
      }

      // Move to next fragment
      this.fragIdx++;
      this.pageIdx = 0;
    }

    return null;
  }

  async next(): Promise<RowBatch | null> {
    const batch = await this.nextColumnar();
    if (!batch) return null;
    return materializeRows(batch);
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
 * Supports OR groups: each group is AND-connected, groups are OR'd (unioned).
 */
function scanFilterIndices(
  decoded: Map<string, DecodedValue[]>,
  columns: ColumnMeta[],
  filters: QueryDescriptor["filters"],
  rowCount: number,
  wasm: WasmEngine,
  filterGroups?: FilterOp[][],
): Uint32Array {
  const colTypes = new Map(columns.map(c => [c.name, c.dtype]));

  // Apply AND filters first
  let indices = applyAndFilters(decoded, colTypes, filters, rowCount, wasm);

  // Apply OR groups: evaluate each group independently, union results
  if (filterGroups && filterGroups.length > 0) {
    let orResult: Uint32Array | null = null;
    for (const group of filterGroups) {
      const groupResult = applyAndFilters(decoded, colTypes, group, rowCount, wasm);
      if (groupResult.length === 0) continue;
      orResult = orResult ? wasmUnion(orResult, groupResult, wasm) : groupResult;
    }
    if (!orResult) return new Uint32Array(0);
    // Intersect OR result with AND result (AND filters + OR groups)
    indices = indices ? wasmIntersect(indices, orResult, wasm) : orResult;
  }

  return indices ?? identityIndices(rowCount);
}

/** Apply AND-connected filters, returning matching row indices. */
function applyAndFilters(
  decoded: Map<string, DecodedValue[]>,
  colTypes: Map<string, string>,
  filters: FilterOp[],
  rowCount: number,
  wasm: WasmEngine,
): Uint32Array {
  let indices: Uint32Array | null = null;

  for (const filter of filters) {
    const dtype = colTypes.get(filter.column);
    const values = decoded.get(filter.column);
    if (!values || values.length === 0) {
      return new Uint32Array(0);
    }

    const isRangeOp = filter.op === "between" || filter.op === "not_between";
    const isCompoundOp = filter.op === "in" || filter.op === "not_in" || isRangeOp || filter.op === "like" || filter.op === "not_like";
    const wasmOp = !isCompoundOp ? filterOpToWasm(filter.op) : -1;

    // Try WASM SIMD path for numeric scalar filters
    if (wasmOp >= 0 && typeof filter.value === "number" &&
        (dtype === "float64" || dtype === "float32" || dtype === "int32" || dtype === "int64")) {
      const filterResult = wasmFilterNumeric(
        values, dtype, wasmOp, filter.value, rowCount, wasm,
      );
      if (filterResult) {
        indices = indices ? wasmIntersect(indices, filterResult, wasm) : filterResult;
        if (indices.length === 0) return indices;
        continue;
      }
    }

    // Try WASM BETWEEN/NOT BETWEEN path for numeric range filters
    if ((filter.op === "between" || filter.op === "not_between") &&
        Array.isArray(filter.value) && filter.value.length === 2 &&
        typeof filter.value[0] === "number" && typeof filter.value[1] === "number" &&
        (dtype === "float64" || dtype === "float32" || dtype === "int32" || dtype === "int64")) {
      const filterResult = wasmFilterRange(
        values, dtype, filter.value[0], filter.value[1], rowCount, wasm, filter.op === "not_between",
      );
      if (filterResult) {
        indices = indices ? wasmIntersect(indices, filterResult, wasm) : filterResult;
        if (indices.length === 0) return indices;
        continue;
      }
    }

    // Try WASM LIKE/NOT LIKE path for string columns
    if ((filter.op === "like" || filter.op === "not_like") &&
        typeof filter.value === "string" &&
        (dtype === "string" || dtype === "utf8" || dtype === "binary")) {
      const filterResult = wasmFilterLike(
        values, filter.value, filter.op === "not_like", rowCount, wasm,
      );
      if (filterResult) {
        indices = indices ? wasmIntersect(indices, filterResult, wasm) : filterResult;
        if (indices.length === 0) return indices;
        continue;
      }
    }

    // JS fallback: evaluate filter on current index set
    const src = indices ?? identityIndices(rowCount);
    const kept: number[] = [];
    for (const idx of src) {
      const v = values[idx];
      if (filter.op === "is_null") { if (v === null || v === undefined) kept.push(idx); continue; }
      if (filter.op === "is_not_null") { if (v !== null && v !== undefined) kept.push(idx); continue; }
      if (v !== null && v !== undefined && matchesFilter(v as Row[string], filter)) {
        kept.push(idx);
      }
    }
    indices = new Uint32Array(kept);
    if (indices.length === 0) return indices;
  }

  return indices ?? identityIndices(rowCount);
}

/** Run WASM SIMD filter on decoded numeric values (f64, i32, i64). */
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
      const dataPtr = wasm.exports.alloc(rowCount * 8);
      if (!dataPtr) return null;
      const dst = new Float64Array(wasm.exports.memory.buffer, dataPtr, rowCount);
      for (let i = 0; i < rowCount; i++) {
        dst[i] = (values[i] as number) ?? 0;
      }
      const outPtr = wasm.exports.alloc(rowCount * 4);
      if (!outPtr) return null;
      const count = wasm.exports.filterFloat64Buffer(
        dataPtr, rowCount, op, filterValue, outPtr, rowCount,
      );
      return new Uint32Array(wasm.exports.memory.buffer.slice(outPtr, outPtr + count * 4));
    }

    if (dtype === "int64") {
      const dataPtr = wasm.exports.alloc(rowCount * 8);
      if (!dataPtr) return null;
      const dst = new BigInt64Array(wasm.exports.memory.buffer, dataPtr, rowCount);
      for (let i = 0; i < rowCount; i++) {
        const v = values[i];
        dst[i] = typeof v === "bigint" ? v : BigInt((v as number) ?? 0);
      }
      const outPtr = wasm.exports.alloc(rowCount * 4);
      if (!outPtr) return null;
      const count = wasm.exports.filterInt64Buffer(
        dataPtr, rowCount, op, BigInt(filterValue), outPtr, rowCount,
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

/** Run WASM BETWEEN (range) filter on decoded numeric values. */
function wasmFilterRange(
  values: DecodedValue[], dtype: string, low: number, high: number,
  rowCount: number, wasm: WasmEngine, negate = false,
): Uint32Array | null {
  try {
    wasm.exports.resetHeap();

    if (dtype === "float64" || dtype === "float32") {
      const dataPtr = wasm.exports.alloc(rowCount * 8);
      if (!dataPtr) return null;
      const dst = new Float64Array(wasm.exports.memory.buffer, dataPtr, rowCount);
      for (let i = 0; i < rowCount; i++) dst[i] = (values[i] as number) ?? 0;
      const outPtr = wasm.exports.alloc(rowCount * 4);
      if (!outPtr) return null;
      const fn = negate ? wasm.exports.filterFloat64NotRange : wasm.exports.filterFloat64Range;
      const count = fn(dataPtr, rowCount, low, high, outPtr, rowCount);
      return new Uint32Array(wasm.exports.memory.buffer.slice(outPtr, outPtr + count * 4));
    }

    if (dtype === "int64") {
      const dataPtr = wasm.exports.alloc(rowCount * 8);
      if (!dataPtr) return null;
      const dst = new BigInt64Array(wasm.exports.memory.buffer, dataPtr, rowCount);
      for (let i = 0; i < rowCount; i++) {
        const v = values[i];
        dst[i] = typeof v === "bigint" ? v : BigInt((v as number) ?? 0);
      }
      const outPtr = wasm.exports.alloc(rowCount * 4);
      if (!outPtr) return null;
      const fn = negate ? wasm.exports.filterInt64NotRange : wasm.exports.filterInt64Range;
      const count = fn(dataPtr, rowCount, BigInt(low), BigInt(high), outPtr, rowCount);
      return new Uint32Array(wasm.exports.memory.buffer.slice(outPtr, outPtr + count * 4));
    }

    if (dtype === "int32") {
      const dataPtr = wasm.exports.alloc(rowCount * 4);
      if (!dataPtr) return null;
      const dst = new Int32Array(wasm.exports.memory.buffer, dataPtr, rowCount);
      for (let i = 0; i < rowCount; i++) dst[i] = (values[i] as number) ?? 0;
      const outPtr = wasm.exports.alloc(rowCount * 4);
      if (!outPtr) return null;
      const fn = negate ? wasm.exports.filterInt32NotRange : wasm.exports.filterInt32Range;
      const count = fn(dataPtr, rowCount, low, high, outPtr, rowCount);
      return new Uint32Array(wasm.exports.memory.buffer.slice(outPtr, outPtr + count * 4));
    }

    return null;
  } catch {
    return null;
  }
}

/** Filter string column with SQL LIKE pattern using WASM. */
function wasmFilterLike(
  values: DecodedValue[],
  pattern: string,
  negated: boolean,
  rowCount: number,
  wasm: WasmEngine,
): Uint32Array | null {
  try {
    wasm.exports.resetHeap();
    const encoder = _textEncoder;

    // Build offsets array and packed string data
    const offsets = new Uint32Array(rowCount + 1);
    let totalBytes = 0;
    const encodedStrings: Uint8Array[] = [];
    for (let i = 0; i < rowCount; i++) {
      offsets[i] = totalBytes;
      const v = values[i];
      const encoded = typeof v === "string" ? encoder.encode(v) : new Uint8Array(0);
      encodedStrings.push(encoded);
      totalBytes += encoded.length;
    }
    offsets[rowCount] = totalBytes;

    // Allocate WASM memory: offsets + string data + pattern + output
    const offsetsPtr = wasm.exports.alloc(offsets.byteLength);
    if (!offsetsPtr) return null;
    new Uint32Array(wasm.exports.memory.buffer, offsetsPtr, offsets.length).set(offsets);

    const dataPtr = wasm.exports.alloc(totalBytes || 1);
    if (!dataPtr) return null;
    const dataDst = new Uint8Array(wasm.exports.memory.buffer, dataPtr, totalBytes);
    let offset = 0;
    for (const encoded of encodedStrings) {
      dataDst.set(encoded, offset);
      offset += encoded.length;
    }

    const patternBytes = encoder.encode(pattern);
    const patternPtr = wasm.exports.alloc(patternBytes.length || 1);
    if (!patternPtr) return null;
    new Uint8Array(wasm.exports.memory.buffer, patternPtr, patternBytes.length).set(patternBytes);

    const outPtr = wasm.exports.alloc(rowCount * 4);
    if (!outPtr) return null;

    const count = wasm.exports.filterStringLike(
      dataPtr, offsetsPtr, rowCount,
      patternPtr, patternBytes.length,
      negated ? 1 : 0,
      outPtr, rowCount,
    );
    return new Uint32Array(wasm.exports.memory.buffer.slice(outPtr, outPtr + count * 4));
  } catch {
    return null;
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

/** Union two sorted index arrays using WASM. */
function wasmUnion(a: Uint32Array, b: Uint32Array, wasm: WasmEngine): Uint32Array {
  try {
    wasm.exports.resetHeap();
    const aPtr = wasm.exports.alloc(a.byteLength);
    const bPtr = wasm.exports.alloc(b.byteLength);
    const outPtr = wasm.exports.alloc((a.length + b.length) * 4);
    if (!aPtr || !bPtr || !outPtr) {
      // Fallback: JS union
      const set = new Set(a);
      for (const v of b) set.add(v);
      return new Uint32Array([...set].sort((x, y) => x - y));
    }
    new Uint32Array(wasm.exports.memory.buffer, aPtr, a.length).set(a);
    new Uint32Array(wasm.exports.memory.buffer, bPtr, b.length).set(b);
    const count = wasm.exports.unionIndices(
      aPtr, a.length, bPtr, b.length, outPtr, a.length + b.length,
    );
    return new Uint32Array(wasm.exports.memory.buffer.slice(outPtr, outPtr + count * 4));
  } catch {
    const set = new Set(a);
    for (const v of b) set.add(v);
    return new Uint32Array([...set].sort((x, y) => x - y));
  }
}

// ---------------------------------------------------------------------------
// Shared page decode helper
// ---------------------------------------------------------------------------

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
      let key: string;
      if (win.partitionBy.length > 0) {
        key = "";
        for (let p = 0; p < win.partitionBy.length; p++) {
          if (p > 0) key += "\x00";
          const v = rows[i][win.partitionBy[p]];
          key += v === null || v === undefined ? NULL_SENTINEL : String(v);
        }
      } else {
        key = "__all__";
      }
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
        const lagCol = win.column ?? (orderBy.length > 0 ? orderBy[0].column : "");
        for (let i = 0; i < indices.length; i++) {
          const srcIdx = i - offset;
          if (srcIdx >= 0 && srcIdx < indices.length && lagCol) {
            rows[indices[i]][alias] = rows[indices[srcIdx]][lagCol] as Row[string];
          } else {
            rows[indices[i]][alias] = defaultVal as Row[string];
          }
        }
        break;
      }

      case "lead": {
        const offset = win.args?.offset ?? 1;
        const defaultVal = win.args?.default_ ?? null;
        const leadCol = win.column ?? (orderBy.length > 0 ? orderBy[0].column : "");
        for (let i = 0; i < indices.length; i++) {
          const srcIdx = i + offset;
          if (srcIdx >= 0 && srcIdx < indices.length && leadCol) {
            rows[indices[i]][alias] = rows[indices[srcIdx]][leadCol] as Row[string];
          } else {
            rows[indices[i]][alias] = defaultVal as Row[string];
          }
        }
        break;
      }

      case "sum": case "avg": case "min": case "max": case "count": {
        const col = win.column ?? (orderBy.length > 0 ? orderBy[0].column : "");
        this.applyAggregateWindow(rows, indices, win, col);
        break;
      }
    }
  }

  private applyAggregateWindow(rows: Row[], indices: number[], win: WindowSpec, col: string): void {
    const { fn, alias, frame } = win;
    const frameStart = frame?.start ?? "unbounded";
    const frameEnd = frame?.end ?? "current";

    // Fast path: "unbounded preceding ... current row" uses O(n) running accumulators
    if (frameStart === "unbounded" && frameEnd === "current") {
      let runSum = 0, runCount = 0, runMin = Infinity, runMax = -Infinity;
      for (let i = 0; i < indices.length; i++) {
        const val = col === "*" ? 1 : rows[indices[i]][col];
        if (val !== null && val !== undefined) {
          const n = typeof val === "number" ? val : typeof val === "bigint" ? Number(val) : 0;
          runSum += n;
          runCount++;
          if (n < runMin) runMin = n;
          if (n > runMax) runMax = n;
        }
        switch (fn) {
          case "sum": rows[indices[i]][alias] = runCount === 0 ? null : runSum; break;
          case "avg": rows[indices[i]][alias] = runCount === 0 ? null : runSum / runCount; break;
          case "min": rows[indices[i]][alias] = runMin === Infinity ? null : runMin; break;
          case "max": rows[indices[i]][alias] = runMax === -Infinity ? null : runMax; break;
          case "count": rows[indices[i]][alias] = runCount; break;
        }
      }
      return;
    }

    // General path for custom frames
    for (let i = 0; i < indices.length; i++) {
      let start: number, end: number;

      if (frameStart === "unbounded") start = 0;
      else if (frameStart === "current") start = i;
      else start = Math.max(0, i + (frameStart as number));

      if (frameEnd === "unbounded") end = indices.length - 1;
      else if (frameEnd === "current") end = i;
      else end = Math.min(indices.length - 1, i + (frameEnd as number));

      let sum = 0, count = 0, min = Infinity, max = -Infinity;
      for (let j = start; j <= end; j++) {
        const val = col === "*" ? 1 : rows[indices[j]][col];
        if (val === null || val === undefined) continue;
        const n = typeof val === "number" ? val : typeof val === "bigint" ? Number(val) : 0;
        sum += n;
        count++;
        if (n < min) min = n;
        if (n > max) max = n;
      }

      switch (fn) {
        case "sum": rows[indices[i]][alias] = count === 0 ? null : sum; break;
        case "avg": rows[indices[i]][alias] = count === 0 ? null : sum / count; break;
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

  async nextColumnar(): Promise<ColumnarBatch | null> {
    if (!this.upstream.nextColumnar) {
      const rows = await this.next();
      if (!rows || rows.length === 0) return null;
      const columns = new Map<string, DecodedValue[]>();
      for (const name of Object.keys(rows[0])) {
        columns.set(name, rows.map(r => (r[name] ?? null) as DecodedValue));
      }
      return { columns, rowCount: rows.length };
    }

    while (true) {
      const batch = await this.upstream.nextColumnar();
      if (!batch) return null;

      const indices = batch.selection ?? identityIndices(batch.rowCount);
      const kept: number[] = [];
      const cols = this.columns.length > 0 ? this.columns : Array.from(batch.columns.keys());

      for (const idx of indices) {
        let key = "";
        for (let g = 0; g < cols.length; g++) {
          if (g > 0) key += "\x00";
          const vals = batch.columns.get(cols[g]);
          const v = vals ? vals[idx] : null;
          key += v === null || v === undefined ? NULL_SENTINEL : String(v);
        }
        if (!this.seen.has(key)) {
          this.seen.add(key);
          kept.push(idx);
        }
      }
      if (kept.length > 0) {
        return { columns: batch.columns, rowCount: batch.rowCount, selection: new Uint32Array(kept) };
      }
    }
  }

  async next(): Promise<RowBatch | null> {
    while (true) {
      const batch = await this.upstream.next();
      if (!batch) return null;

      const unique: Row[] = [];
      for (const row of batch) {
        let key = "";
        const keyCols = this.columns.length > 0 ? this.columns : Object.keys(row);
        for (let g = 0; g < keyCols.length; g++) {
          if (g > 0) key += "\x00";
          const v = row[keyCols[g]];
          key += v === null || v === undefined ? NULL_SENTINEL : String(v);
        }
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
    const keys = Object.keys(row).sort();
    let result = "";
    for (let i = 0; i < keys.length; i++) {
      if (i > 0) result += "\x00";
      const v = row[keys[i]];
      result += keys[i] + "=" + (v === null || v === undefined ? NULL_SENTINEL : String(v));
    }
    return result;
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
  private filterGroups?: FilterOp[][];

  constructor(upstream: Operator, filters: QueryDescriptor["filters"], filterGroups?: FilterOp[][]) {
    this.upstream = upstream;
    this.filters = filters;
    this.filterGroups = filterGroups;
  }

  private matchesRow(row: Row): boolean {
    return rowPassesFilters(row, this.filters, this.filterGroups);
  }

  async nextColumnar(): Promise<ColumnarBatch | null> {
    if (!this.upstream.nextColumnar) {
      // Upstream doesn't support columnar — fall back to row-based and re-columnize
      const rows = await this.next();
      if (!rows || rows.length === 0) return null;
      const columns = new Map<string, DecodedValue[]>();
      const colNames = Object.keys(rows[0]);
      for (const name of colNames) {
        columns.set(name, rows.map(r => (r[name] ?? null) as DecodedValue));
      }
      return { columns, rowCount: rows.length };
    }

    while (true) {
      const batch = await this.upstream.nextColumnar();
      if (!batch) return null;

      // Apply filters on the columnar batch — narrow the selection vector
      const srcIndices = batch.selection ?? identityIndices(batch.rowCount);
      const kept: number[] = [];

      for (const idx of srcIndices) {
        let andPass = true;
        for (const f of this.filters) {
          const vals = batch.columns.get(f.column);
          const v = vals ? vals[idx] : null;
          if (f.op === "is_null") { if (v !== null && v !== undefined) { andPass = false; break; } continue; }
          if (f.op === "is_not_null") { if (v === null || v === undefined) { andPass = false; break; } continue; }
          if (v === null || v === undefined || !matchesFilter(v as Row[string], f)) {
            andPass = false;
            break;
          }
        }
        if (!andPass) continue;

        if (this.filterGroups && this.filterGroups.length > 0) {
          const orPass = this.filterGroups.some(group =>
            group.every(f => {
              const vals = batch.columns.get(f.column);
              const v = vals ? vals[idx] : null;
              if (f.op === "is_null") return v === null || v === undefined;
              if (f.op === "is_not_null") return v !== null && v !== undefined;
              return v !== null && v !== undefined && matchesFilter(v as Row[string], f);
            }),
          );
          if (!orPass) continue;
        }

        kept.push(idx);
      }

      if (kept.length > 0) {
        return { columns: batch.columns, rowCount: batch.rowCount, selection: new Uint32Array(kept) };
      }
      // Empty batch after filtering — pull next
    }
  }

  async next(): Promise<RowBatch | null> {
    while (true) {
      const batch = await this.upstream.next();
      if (!batch) return null;

      const filtered: Row[] = [];
      for (const row of batch) {
        if (this.matchesRow(row)) filtered.push(row);
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

  async nextColumnar(): Promise<ColumnarBatch | null> {
    if (this.remaining <= 0) return null;
    if (!this.upstream.nextColumnar) {
      const rows = await this.next();
      if (!rows || rows.length === 0) return null;
      const columns = new Map<string, DecodedValue[]>();
      for (const name of Object.keys(rows[0])) {
        columns.set(name, rows.map(r => (r[name] ?? null) as DecodedValue));
      }
      return { columns, rowCount: rows.length };
    }

    while (true) {
      const batch = await this.upstream.nextColumnar();
      if (!batch) return null;

      let sel = batch.selection ?? identityIndices(batch.rowCount);

      // Handle offset: skip rows
      if (this.skipped < this.offset) {
        const toSkip = this.offset - this.skipped;
        if (toSkip >= sel.length) {
          this.skipped += sel.length;
          continue;
        }
        sel = sel.subarray(toSkip);
        this.skipped = this.offset;
      }

      // Apply limit
      if (sel.length > this.remaining) {
        sel = sel.slice(0, this.remaining);
      }
      this.remaining -= sel.length;

      if (sel.length > 0) {
        return { columns: batch.columns, rowCount: batch.rowCount, selection: sel };
      }
      return null;
    }
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

  async nextColumnar(): Promise<ColumnarBatch | null> {
    if (this.upstream.nextColumnar) {
      const batch = await this.upstream.nextColumnar();
      if (!batch) return null;
      const projected = new Map<string, DecodedValue[]>();
      for (const [name, vals] of batch.columns) {
        if (this.keep.has(name)) projected.set(name, vals);
      }
      return { columns: projected, rowCount: batch.rowCount, selection: batch.selection };
    }
    // Fall back to row-based
    const rows = await this.next();
    if (!rows || rows.length === 0) return null;
    const columns = new Map<string, DecodedValue[]>();
    for (const name of this.keep) {
      columns.set(name, rows.map(r => (r[name] ?? null) as DecodedValue));
    }
    return { columns, rowCount: rows.length };
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

    // Use columnar path if upstream supports it — avoids Row[] materialization during aggregation
    if (this.upstream.nextColumnar) {
      let merged: PartialAgg | null = null;
      while (true) {
        const batch = await this.upstream.nextColumnar();
        if (!batch) break;
        const partial = computePartialAggColumnar(batch, this.query);
        merged = merged ? mergePartialAggs([merged, partial]) : partial;
      }
      if (!merged) return finalizePartialAgg({ states: [] }, this.query);
      return finalizePartialAgg(merged, this.query);
    }

    // Row-based fallback
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

  /** Drain all upstream batches into the heap, using columnar path if available. */
  private async drainIntoHeap(): Promise<Row[]> {
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

    const pushRow = (row: Row) => {
      if (heap.length < k) { heap.push(row); siftUp(heap.length - 1); }
      else if (heap.length > 0 && shouldReplace(row)) { heap[0] = row; siftDown(heap, 0); }
    };

    // Use columnar path if available — only materialize rows that would enter the heap
    if (this.upstream.nextColumnar) {
      while (true) {
        const batch = await this.upstream.nextColumnar();
        if (!batch) break;
        const indices = batch.selection ?? identityIndices(batch.rowCount);
        const colNames = Array.from(batch.columns.keys());
        const sortVals = batch.columns.get(col);
        for (const idx of indices) {
          // Fast reject: if heap is full and this value can't beat the root, skip materialization
          if (sortVals && heap.length >= k) {
            const nv = sortVals[idx] as Row[string];
            const rv = heap[0][col];
            if (nv === null) continue;
            if (rv !== null && (desc ? nv <= rv : nv >= rv)) continue;
          }
          const row: Row = {};
          for (const name of colNames) {
            row[name] = (batch.columns.get(name)![idx] as Row[string]) ?? null;
          }
          pushRow(row);
        }
      }
    } else {
      while (true) {
        const batch = await this.upstream.next();
        if (!batch) break;
        for (const row of batch) pushRow(row);
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
    return result;
  }

  async next(): Promise<RowBatch | null> {
    if (this.consumed) return null;
    this.consumed = true;
    const result = await this.drainIntoHeap();
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

    // Collect all rows — use columnar path if available
    const allRows: Row[] = [];
    if (this.upstream.nextColumnar) {
      while (true) {
        const batch = await this.upstream.nextColumnar();
        if (!batch) break;
        for (const row of materializeRows(batch)) allRows.push(row);
      }
    } else {
      while (true) {
        const batch = await this.upstream.next();
        if (!batch) break;
        for (const row of batch) allRows.push(row);
      }
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
  const colMap = new Map(columns.map(c => [c.name, c]));

  // Validate filter is numeric with scalar/range op
  const isValidFilter = (f: FilterOp): boolean => {
    if (f.op === "in" || f.op === "not_in" || f.op === "like" || f.op === "not_like" || f.op === "is_null" || f.op === "is_not_null") return false;
    const fc = colMap.get(f.column);
    if (!fc) return false;
    if (fc.dtype !== "float64" && fc.dtype !== "int32" && fc.dtype !== "int64") return false;
    if (fc.pages.some(p => p.encoding)) return false;
    if (f.op === "between" || f.op === "not_between") {
      if (!Array.isArray(f.value) || f.value.length !== 2) return false;
      if (typeof f.value[0] !== "number" || typeof f.value[1] !== "number") return false;
    } else {
      if (typeof f.value !== "number") return false;
    }
    return true;
  };

  // AND filters must all be valid
  for (const f of query.filters) {
    if (!isValidFilter(f)) return false;
  }

  // OR filter groups: each group's filters must all be valid
  if (query.filterGroups) {
    for (const group of query.filterGroups) {
      for (const f of group) {
        if (!isValidFilter(f)) return false;
      }
    }
  }

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
    const filters = this.query.filters;
    const filterGroups = this.query.filterGroups;
    const hasFilters = filters.length > 0 || (filterGroups && filterGroups.length > 0);
    // Accumulator per aggregate
    const acc: { sum: number; count: number; min: number; max: number }[] =
      aggregates.map(() => ({ sum: 0, count: 0, min: Infinity, max: -Infinity }));

    const scanStart = Date.now();
    for (const frag of this.fragments) {
      const colMap = new Map(frag.columns.map(c => [c.name, c]));

      // Separate filter columns from aggregate-only columns
      const filterColNames = new Set(filters.map(f => f.column));
      if (filterGroups) for (const g of filterGroups) for (const f of g) filterColNames.add(f.column);
      const aggOnlyColNames = new Set<string>();
      for (const agg of aggregates) {
        if (agg.column !== "*" && !filterColNames.has(agg.column)) aggOnlyColNames.add(agg.column);
      }

      const firstCol = frag.columns[0];
      if (!firstCol) continue;
      const pageCount = firstCol.pages.length;

      for (let pi = 0; pi < pageCount; pi++) {
        if (canSkipPageMultiCol(frag.columns, pi, filters, filterGroups)) {
          this.pagesSkipped++;
          continue;
        }

        // Phase 1: Read filter columns (+ any aggregate columns that are also filter columns)
        const pageBuffers = new Map<string, ArrayBuffer>();
        for (const colName of filterColNames) {
          const col = colMap.get(colName);
          if (!col || !col.pages[pi]) continue;
          const buf = await frag.readPage(col, col.pages[pi]);
          this.bytesRead += buf.byteLength;
          pageBuffers.set(colName, buf);
        }

        // Compute matching indices if filters exist
        let matchCount = -1; // -1 = no filter, use full buffer
        let indicesPtr = 0;
        if (hasFilters) {
          this.wasm.exports.resetHeap();

          // Evaluate AND filters
          let currentIndices: Uint32Array | null = null;
          if (filters.length > 0) {
            currentIndices = this.evalAndFilters(filters, colMap, pageBuffers);
            if (currentIndices.length === 0) continue;
          }

          // Evaluate OR filter groups: each group independently, union results
          if (filterGroups && filterGroups.length > 0) {
            let orResult: Uint32Array | null = null;
            for (const group of filterGroups) {
              const groupResult = this.evalAndFilters(group, colMap, pageBuffers);
              if (groupResult.length === 0) continue;
              orResult = orResult ? wasmUnion(orResult, groupResult, this.wasm) : groupResult;
            }
            if (!orResult) continue; // all OR groups empty
            // Intersect OR result with AND result
            currentIndices = currentIndices ? wasmIntersect(currentIndices, orResult, this.wasm) : orResult;
          }

          if (!currentIndices || currentIndices.length === 0) continue;

          // Copy indices to WASM for indexed aggregates
          this.wasm.exports.resetHeap();
          indicesPtr = this.wasm.exports.alloc(currentIndices.byteLength);
          if (indicesPtr) {
            new Uint32Array(this.wasm.exports.memory.buffer, indicesPtr, currentIndices.length).set(currentIndices);
          }
          matchCount = currentIndices.length;
        }

        // Phase 2: Read aggregate-only columns (skipped if filter eliminated all rows)
        if (!hasFilters || matchCount > 0) {
          for (const colName of aggOnlyColNames) {
            if (pageBuffers.has(colName)) continue; // already read (also a filter column)
            const col = colMap.get(colName);
            if (!col || !col.pages[pi]) continue;
            const buf = await frag.readPage(col, col.pages[pi]);
            this.bytesRead += buf.byteLength;
            pageBuffers.set(colName, buf);
          }
        }

        // Aggregate per column
        for (let ai = 0; ai < aggregates.length; ai++) {
          const agg = aggregates[ai];
          if (agg.column === "*") {
            acc[ai].count += hasFilters ? matchCount : firstCol.pages[pi].rowCount;
            continue;
          }

          const col = colMap.get(agg.column);
          const buf = pageBuffers.get(agg.column);
          if (!col || !buf) continue;

          if (!hasFilters) {
            // Unfiltered: use full-buffer SIMD aggregates
            const count = buf.byteLength >> 3;
            acc[ai].count += count;
            if (col.dtype === "float64") {
              if (agg.fn === "sum" || agg.fn === "avg") acc[ai].sum += this.wasm.sumFloat64(buf);
              if (agg.fn === "min") { const v = this.wasm.minFloat64(buf); if (v < acc[ai].min) acc[ai].min = v; }
              if (agg.fn === "max") { const v = this.wasm.maxFloat64(buf); if (v > acc[ai].max) acc[ai].max = v; }
            } else if (col.dtype === "int64") {
              if (agg.fn === "sum" || agg.fn === "avg") acc[ai].sum += Number(this.wasm.sumInt64(buf));
              if (agg.fn === "min") { const v = Number(this.wasm.minInt64(buf)); if (v < acc[ai].min) acc[ai].min = v; }
              if (agg.fn === "max") { const v = Number(this.wasm.maxInt64(buf)); if (v > acc[ai].max) acc[ai].max = v; }
            }
          } else {
            // Filtered: use indexed aggregates on matching rows only
            acc[ai].count += matchCount;
            const dataPtr = this.wasm.exports.alloc(buf.byteLength);
            if (!dataPtr) continue;
            new Uint8Array(this.wasm.exports.memory.buffer, dataPtr, buf.byteLength).set(new Uint8Array(buf));

            if (col.dtype === "float64") {
              if (agg.fn === "sum" || agg.fn === "avg") acc[ai].sum += this.wasm.exports.sumFloat64Indexed(dataPtr, indicesPtr, matchCount);
              if (agg.fn === "min") { const v = this.wasm.exports.minFloat64Indexed(dataPtr, indicesPtr, matchCount); if (v < acc[ai].min) acc[ai].min = v; }
              if (agg.fn === "max") { const v = this.wasm.exports.maxFloat64Indexed(dataPtr, indicesPtr, matchCount); if (v > acc[ai].max) acc[ai].max = v; }
            } else if (col.dtype === "int64") {
              if (agg.fn === "sum" || agg.fn === "avg") acc[ai].sum += Number(this.wasm.exports.sumInt64Indexed(dataPtr, indicesPtr, matchCount));
              if (agg.fn === "min") { const v = Number(this.wasm.exports.minInt64Indexed(dataPtr, indicesPtr, matchCount)); if (v < acc[ai].min) acc[ai].min = v; }
              if (agg.fn === "max") { const v = Number(this.wasm.exports.maxInt64Indexed(dataPtr, indicesPtr, matchCount)); if (v > acc[ai].max) acc[ai].max = v; }
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
        case "sum": row[alias] = acc[i].count === 0 ? null : acc[i].sum; break;
        case "avg": row[alias] = acc[i].count === 0 ? null : acc[i].sum / acc[i].count; break;
        case "min": row[alias] = acc[i].count === 0 ? null : acc[i].min; break;
        case "max": row[alias] = acc[i].count === 0 ? null : acc[i].max; break;
        case "count": row[alias] = acc[i].count; break;
      }
    }

    return [row];
  }

  /** Evaluate AND-connected filters on page buffers, returning matching row indices. */
  private evalAndFilters(
    filters: FilterOp[],
    colMap: Map<string, ColumnMeta>,
    pageBuffers: Map<string, ArrayBuffer>,
  ): Uint32Array {
    let currentIndices: Uint32Array | null = null;

    for (const f of filters) {
      const col = colMap.get(f.column);
      const buf = pageBuffers.get(f.column);
      if (!col || !buf) return new Uint32Array(0);
      if (col.dtype !== "float64" && col.dtype !== "int32" && col.dtype !== "int64") return new Uint32Array(0);
      const elemSize = col.dtype === "int32" ? 4 : 8;
      const rowCount = buf.byteLength / elemSize;

      this.wasm.exports.resetHeap();
      const dataPtr = this.wasm.exports.alloc(buf.byteLength);
      if (!dataPtr) return new Uint32Array(0);
      new Uint8Array(this.wasm.exports.memory.buffer, dataPtr, buf.byteLength).set(new Uint8Array(buf));
      const outPtr = this.wasm.exports.alloc(rowCount * 4);
      if (!outPtr) return new Uint32Array(0);

      let count: number;
      if ((f.op === "between" || f.op === "not_between") && Array.isArray(f.value) && f.value.length === 2) {
        const isNotBetween = f.op === "not_between";
        if (col.dtype === "float64") {
          count = isNotBetween
            ? this.wasm.exports.filterFloat64NotRange(dataPtr, rowCount, f.value[0] as number, f.value[1] as number, outPtr, rowCount)
            : this.wasm.exports.filterFloat64Range(dataPtr, rowCount, f.value[0] as number, f.value[1] as number, outPtr, rowCount);
        } else if (col.dtype === "int64") {
          count = isNotBetween
            ? this.wasm.exports.filterInt64NotRange(dataPtr, rowCount, BigInt(f.value[0] as number), BigInt(f.value[1] as number), outPtr, rowCount)
            : this.wasm.exports.filterInt64Range(dataPtr, rowCount, BigInt(f.value[0] as number), BigInt(f.value[1] as number), outPtr, rowCount);
        } else {
          count = isNotBetween
            ? this.wasm.exports.filterInt32NotRange(dataPtr, rowCount, f.value[0] as number, f.value[1] as number, outPtr, rowCount)
            : this.wasm.exports.filterInt32Range(dataPtr, rowCount, f.value[0] as number, f.value[1] as number, outPtr, rowCount);
        }
      } else {
        const wasmOp = filterOpToWasm(f.op);
        if (wasmOp < 0) return new Uint32Array(0);
        if (col.dtype === "float64") {
          count = this.wasm.exports.filterFloat64Buffer(dataPtr, rowCount, wasmOp, f.value as number, outPtr, rowCount);
        } else if (col.dtype === "int64") {
          count = this.wasm.exports.filterInt64Buffer(dataPtr, rowCount, wasmOp, BigInt(f.value as number), outPtr, rowCount);
        } else {
          count = this.wasm.exports.filterInt32Buffer(dataPtr, rowCount, wasmOp, f.value as number, outPtr, rowCount);
        }
      }
      const filterResult = new Uint32Array(this.wasm.exports.memory.buffer.slice(outPtr, outPtr + count * 4));

      if (currentIndices) {
        currentIndices = wasmIntersect(currentIndices, filterResult, this.wasm);
        if (currentIndices.length === 0) return currentIndices;
      } else {
        currentIndices = filterResult;
        if (currentIndices.length === 0) return currentIndices;
      }
    }

    return currentIndices ?? new Uint32Array(0);
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
  private mergeSiftDown: ((i: number) => void) | null = null;
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

    const consumeRow = async (row: Row) => {
      const rowSize = estimateRowSize(row);
      if (currentRunBytes + rowSize > this.memoryBudget && currentRun.length > 0) {
        await flushRun();
      }
      currentRun.push(row);
      currentRunBytes += rowSize;
    };

    // Use columnar path if available
    if (this.upstream.nextColumnar) {
      while (true) {
        const batch = await this.upstream.nextColumnar();
        if (!batch) break;
        for (const row of materializeRows(batch)) await consumeRow(row);
      }
    } else {
      while (true) {
        const batch = await this.upstream.next();
        if (!batch) break;
        for (const row of batch) await consumeRow(row);
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

    for (let i = Math.floor(heap.length / 2) - 1; i >= 0; i--) siftDown(i);

    this.mergeReaders = readers;
    this.mergeHeap = heap;
    this.mergeSiftDown = siftDown;
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
    const siftDown = this.mergeSiftDown!;

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
  const pipeline = assemblePipeline(scan, query, outputColumns, scan.filtersApplied, options);
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
  const filtersApplied = "filtersApplied" in scan && (scan as { filtersApplied: boolean }).filtersApplied;
  // Skip projection when aggregation is active — aggregate output columns
  // (aliases like "count_*", "sum_value") don't match the original table columns,
  // so projecting to table columns would strip all aggregate results.
  const hasAgg = query.aggregates && query.aggregates.length > 0;
  const projectCols = hasAgg ? [] : outputColumns;
  return assemblePipeline(scan, query, projectCols, filtersApplied, options);
}

/** Shared pipeline assembly: filter → subquery → computed → pipe → window → distinct → agg → sort → project */
function assemblePipeline(
  source: Operator,
  query: QueryDescriptor,
  outputColumns: string[],
  filtersApplied: boolean,
  options?: PipelineOptions,
): Operator {
  let pipeline: Operator = source;
  const memBudget = options?.memoryBudgetBytes ?? DEFAULT_MEMORY_BUDGET;

  const hasFilters = query.filters.length > 0 || (query.filterGroups && query.filterGroups.length > 0);
  if (hasFilters && !filtersApplied) {
    pipeline = new FilterOperator(pipeline, query.filters, query.filterGroups);
  }

  if (query.subqueryIn) {
    for (const sq of query.subqueryIn) {
      pipeline = new SubqueryInOperator(pipeline, sq.column, sq.valueSet);
    }
  }

  if (query.computedColumns) {
    pipeline = new ComputedColumnOperator(pipeline, query.computedColumns);
  }

  if (query.pipeStages) {
    for (const stage of query.pipeStages) {
      pipeline = stage(pipeline);
    }
  }

  if (query.windows && query.windows.length > 0) {
    pipeline = new WindowOperator(pipeline, query.windows);
  }

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

  if (outputColumns.length > 0) {
    pipeline = new ProjectOperator(pipeline, outputColumns);
  }

  return pipeline;
}

/**
 * Drain all rows from an operator pipeline.
 */
export async function drainPipeline(pipeline: Operator): Promise<Row[]> {
  // Use columnar path if the pipeline supports it — avoids Row[] materialization in intermediate operators
  if (pipeline.nextColumnar) {
    const rows: Row[] = [];
    while (true) {
      const batch = await pipeline.nextColumnar();
      if (!batch) break;
      // Materialize only at the pipeline exit
      for (const row of materializeRows(batch)) rows.push(row);
    }
    await pipeline.close();
    return rows;
  }

  const rows: Row[] = [];
  while (true) {
    const batch = await pipeline.next();
    if (!batch) break;
    for (const row of batch) rows.push(row);
  }
  await pipeline.close();
  return rows;
}
