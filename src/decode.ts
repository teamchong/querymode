import type { ColumnMeta, PageInfo, Row, DataType } from "./types.js";
import type { QueryDescriptor } from "./client.js";
import type { WasmEngine } from "./wasm-engine.js";
import { decodeParquetColumnChunk, decodePlainValues } from "./parquet-decode.js";
import { decodeLanceV2Utf8 } from "./lance-v2.js";

/** Check if a page can be skipped via min/max stats. */
export function canSkipPage(page: PageInfo, filters: QueryDescriptor["filters"], columnName: string): boolean {
  for (const filter of filters) {
    if (filter.column !== columnName) continue;
    if (filter.op === "is_null" || filter.op === "is_not_null") continue;
    if (page.minValue === undefined || page.maxValue === undefined) continue;

    let { minValue: min, maxValue: max } = page;
    let val = filter.value;
    if (typeof val === "object" && !Array.isArray(val)) continue;

    // Coerce bigint↔number for cross-type comparisons
    if (typeof min === "bigint" && typeof val === "number") val = BigInt(Math.trunc(val as number));
    else if (typeof min === "number" && typeof val === "bigint") { min = BigInt(Math.trunc(min)); max = BigInt(Math.trunc(max as number)); }

    switch (filter.op) {
      case "gt":  if (max <= val) return true; break;
      case "gte": if (max < val) return true; break;
      case "lt":  if (min >= val) return true; break;
      case "lte": if (min > val) return true; break;
      case "eq":  if (val < min || val > max) return true; break;
      case "between": {
        if (!Array.isArray(filter.value) || filter.value.length !== 2) break;
        let lo = filter.value[0];
        let hi = filter.value[1];
        if (typeof min === "bigint" && typeof lo === "number") { lo = BigInt(Math.trunc(lo)); hi = BigInt(Math.trunc(hi as number)); }
        else if (typeof min === "number" && typeof lo === "bigint") { min = BigInt(Math.trunc(min)); max = BigInt(Math.trunc(max as number)); }
        if (max < lo || min > hi) return true;
        break;
      }
      case "not_between": {
        if (!Array.isArray(filter.value) || filter.value.length !== 2) break;
        let lo = filter.value[0];
        let hi = filter.value[1];
        if (typeof min === "bigint" && typeof lo === "number") { lo = BigInt(Math.trunc(lo)); hi = BigInt(Math.trunc(hi as number)); }
        else if (typeof min === "number" && typeof lo === "bigint") { min = BigInt(Math.trunc(min)); max = BigInt(Math.trunc(max as number)); }
        // Skip if all values are within [lo, hi] — NOT BETWEEN would exclude them all
        if (min >= lo && max <= hi) return true;
        break;
      }
    }
  }
  return false;
}

/** Check if an entire fragment can be skipped based on fragment-level min/max stats.
 *  Computes per-column min/max across all pages, then checks if any AND filter
 *  eliminates the entire range. For OR groups, skips only if ALL groups eliminate it. */
export function canSkipFragment(
  meta: { columns: ColumnMeta[] },
  filters: QueryDescriptor["filters"],
  filterGroups?: import("./types.js").FilterOp[][],
): boolean {
  if (filters.length === 0 && (!filterGroups || filterGroups.length === 0)) return false;

  // Compute per-column min/max across all pages
  const colStats = new Map<string, { min: number | bigint | string; max: number | bigint | string }>();
  for (const col of meta.columns) {
    let colMin: number | bigint | string | undefined;
    let colMax: number | bigint | string | undefined;
    for (const page of col.pages) {
      if (page.minValue === undefined || page.maxValue === undefined) {
        colMin = undefined;
        break;
      }
      if (colMin === undefined || page.minValue < colMin) colMin = page.minValue;
      if (colMax === undefined || page.maxValue > colMax) colMax = page.maxValue;
    }
    if (colMin !== undefined && colMax !== undefined) {
      colStats.set(col.name, { min: colMin, max: colMax });
    }
  }

  // Build a synthetic "fragment page" with the aggregated stats and reuse canSkipPage
  for (const f of filters) {
    const stats = colStats.get(f.column);
    if (!stats) continue;
    const syntheticPage: PageInfo = {
      byteOffset: 0n, byteLength: 0, rowCount: 0, nullCount: 0,
      minValue: stats.min, maxValue: stats.max,
    };
    if (canSkipPage(syntheticPage, [f], f.column)) return true;
  }

  // OR groups: skip only if ALL groups eliminate the fragment
  if (filterGroups && filterGroups.length > 0) {
    const allGroupsSkip = filterGroups.every(group => {
      for (const f of group) {
        const stats = colStats.get(f.column);
        if (!stats) continue;
        const syntheticPage: PageInfo = {
          byteOffset: 0n, byteLength: 0, rowCount: 0, nullCount: 0,
          minValue: stats.min, maxValue: stats.max,
        };
        if (canSkipPage(syntheticPage, [f], f.column)) return true;
      }
      return false;
    });
    if (allGroupsSkip) return true;
  }

  return false;
}

/** Assemble rows from column page buffers. WASM required. */
export function assembleRows(
  columnData: Map<string, ArrayBuffer[]>,
  projectedColumns: ColumnMeta[],
  query: QueryDescriptor,
  wasmEngine: WasmEngine,
): Row[] {
  if (query.vectorSearch) {
    return vectorSearch(columnData, projectedColumns, query, wasmEngine);
  }

  const decoded = decodeAllColumns(columnData, projectedColumns, undefined, wasmEngine);
  const firstCol = decoded.values().next().value;
  if (!firstCol) return [];
  const rowCount = firstCol.length;

  const rows: Row[] = [];
  const hasSort = !!query.sortColumn;
  const limit = query.limit ?? Infinity;
  const offset = query.offset ?? 0;
  // Without sort, apply offset+limit as early termination
  const earlyStop = !hasSort ? offset + limit : rowCount;

  for (let i = 0; i < rowCount && rows.length < earlyStop; i++) {
    const row: Row = {};
    for (const col of projectedColumns) {
      const values = decoded.get(col.name);
      row[col.name] = values ? (values[i] as Row[string]) : null;
    }

    // AND filters must all pass
    const andPass = query.filters.every(f => {
      const v = row[f.column];
      if (f.op === "is_null") return v === null || v === undefined;
      if (f.op === "is_not_null") return v !== null && v !== undefined;
      return v !== null && matchesFilter(v, f);
    });
    if (!andPass) continue;

    // OR groups: at least one group must pass
    if (query.filterGroups && query.filterGroups.length > 0) {
      const orPass = query.filterGroups.some(group =>
        group.every(f => {
          const v = row[f.column];
          if (f.op === "is_null") return v === null || v === undefined;
          if (f.op === "is_not_null") return v !== null && v !== undefined;
          return v !== null && matchesFilter(v, f);
        }),
      );
      if (!orPass) continue;
    }

    rows.push(row);
  }

  return applySortAndLimit(rows, query);
}

// --- Column decoding ---

type DecodedValue = number | bigint | string | boolean | Float32Array | null;

/** Decode all projected columns from their page buffers. */
function decodeAllColumns(
  columnData: Map<string, ArrayBuffer[]>,
  columns: ColumnMeta[],
  exclude: string | undefined,
  wasm: WasmEngine,
): Map<string, DecodedValue[]> {
  const result = new Map<string, DecodedValue[]>();
  for (const col of columns) {
    if (col.name === exclude) continue;
    const pages = columnData.get(col.name);
    if (!pages?.length) continue;

    if (col.dtype === "fixed_size_list") {
      result.set(col.name, decodeFixedSizeListPages(pages, col.listDimension ?? 0));
    } else {
      const values: DecodedValue[] = [];
      for (let i = 0; i < pages.length; i++) {
        const pi = col.pages[i];
        const decoded = pi?.encoding
          ? decodeParquetColumnChunk(pages[i], pi.encoding, col.dtype, pi.rowCount, wasm)
          : decodePage(pages[i], col.dtype, pi?.nullCount ?? 0, pi?.rowCount ?? 0, pi?.dataOffsetInPage);
        for (const v of decoded) {
          values.push(v);
        }
      }
      result.set(col.name, values);
    }
  }
  return result;
}

function decodeFixedSizeListPages(pages: ArrayBuffer[], dim: number): Float32Array[] {
  if (dim === 0) return [];
  const result: Float32Array[] = [];
  for (const pageBuf of pages) {
    const view = new DataView(pageBuf);
    const numRows = Math.floor((pageBuf.byteLength >> 2) / dim);
    for (let r = 0; r < numRows; r++) {
      const vec = new Float32Array(dim);
      for (let d = 0; d < dim; d++) vec[d] = view.getFloat32((r * dim + d) * 4, true);
      result.push(vec);
    }
  }
  return result;
}


/** Decode a raw page buffer into typed values. Handles null bitmaps when nullCount > 0.
 * @param dataOffsetInPage - For Lance v2 nullable pages: byte offset where data starts (after bitmap + alignment padding)
 */
export function decodePage(
  buf: ArrayBuffer,
  dtype: string,
  nullCount = 0,
  rowCount = 0,
  dataOffsetInPage?: number,
): (number | bigint | string | null)[] {
  let nulls: Set<number> | null = null;
  if (nullCount > 0 && rowCount > 0) {
    const bitmapBytes = Math.ceil(rowCount / 8);
    if (buf.byteLength < bitmapBytes) return [];
    const bytes = new Uint8Array(buf, 0, bitmapBytes);
    nulls = new Set<number>();
    // Fast path: skip bytes that are 0xFF (all valid) — avoids per-bit work
    for (let b = 0; b < bitmapBytes; b++) {
      const byte = bytes[b];
      if (byte === 0xFF) continue; // all 8 bits valid, skip
      const base = b << 3;
      if (byte === 0x00) {
        // all 8 bits null — add them all
        const end = Math.min(base + 8, rowCount);
        for (let i = base; i < end; i++) nulls.add(i);
      } else {
        // mixed — check each bit
        const end = Math.min(8, rowCount - base);
        for (let bit = 0; bit < end; bit++) {
          if (((byte >> bit) & 1) === 0) nulls.add(base + bit);
        }
      }
    }
    // Lance v2 uses alignment padding between bitmap and data.
    // dataOffsetInPage gives the exact data start; otherwise strip only bitmap bytes.
    const stripBytes = dataOffsetInPage ?? bitmapBytes;
    buf = buf.slice(stripBytes);
  }

  const bytes = new Uint8Array(buf);
  // Lance v2 stores ALL row slots (including zeros at null positions).
  // When dataOffsetInPage is set, decode all rowCount values and mask nulls.
  // Parquet-style packs only non-null values, so decode rowCount - nulls.size.
  const isLanceV2Nullable = dataOffsetInPage !== undefined;
  const numValues = rowCount > 0
    ? (isLanceV2Nullable ? rowCount : rowCount - (nulls?.size ?? 0))
    : Number.MAX_SAFE_INTEGER;

  // For utf8/binary with rowCount known and buffer large enough to contain an offsets array,
  // try Lance v2 format (i64 offsets + string data) first.
  if ((dtype === "utf8" || dtype === "binary") && rowCount > 0 && buf.byteLength >= rowCount * 8) {
    const decodeCount = isLanceV2Nullable ? rowCount : numValues;
    const v2Strings = decodeLanceV2Utf8(buf, decodeCount);
    if (v2Strings.length === decodeCount && v2Strings.every(s => typeof s === "string")) {
      const looksValid = v2Strings.every(s => s.length < buf.byteLength);
      if (looksValid) {
        if (nulls && nulls.size > 0) {
          if (isLanceV2Nullable) {
            // Lance v2: all slots present, just null-mask them
            return v2Strings.map((s, i) => nulls!.has(i) ? null : s);
          }
          // Parquet-style: packed non-null values, interleave with nulls
          const withNulls: (string | null)[] = [];
          let vi = 0;
          for (let i = 0; i < rowCount; i++) {
            withNulls.push(nulls.has(i) ? null : (vi < v2Strings.length ? v2Strings[vi++] : null));
          }
          return withNulls;
        }
        return v2Strings;
      }
    }
  }

  const values = decodePlainValues(bytes, dtype as DataType, numValues) as (number | bigint | string | null)[];

  if (nulls && nulls.size > 0) {
    if (isLanceV2Nullable) {
      // Lance v2: all row slots present, replace null positions with null
      return values.map((v, i) => nulls!.has(i) ? null : v);
    }
    // Parquet-style: packed non-null values, interleave with nulls
    const withNulls: (number | bigint | string | null)[] = [];
    let vi = 0;
    for (let i = 0; i < rowCount; i++) {
      withNulls.push(nulls.has(i) ? null : (vi < values.length ? values[vi++] : null));
    }
    return withNulls;
  }

  return values;
}

// --- Sort / Top-K ---

function applySortAndLimit(rows: Row[], query: QueryDescriptor): Row[] {
  const offset = query.offset ?? 0;
  const limit = query.limit;

  if (!query.sortColumn || rows.length === 0) {
    // No sort — just apply offset + limit
    if (offset > 0 || limit !== undefined) {
      return rows.slice(offset, limit !== undefined ? offset + limit : undefined);
    }
    return rows;
  }

  const col = query.sortColumn;
  const desc = query.sortDirection === "desc";
  const needed = offset + (limit ?? rows.length);

  if (needed < rows.length) {
    const sorted = topK(rows, needed, col, desc);
    return sorted.slice(offset);
  }

  const dir = desc ? -1 : 1;
  rows.sort((a, b) => {
    const av = a[col], bv = b[col];
    if (av === null && bv === null) return 0;
    if (av === null) return 1;
    if (bv === null) return -1;
    return av < bv ? -dir : av > bv ? dir : 0;
  });
  const start = offset;
  const end = limit !== undefined ? offset + limit : undefined;
  return rows.slice(start, end);
}

/** Top-K via max-heap (asc) or min-heap (desc). O(N log K) time, O(K) memory. */
function topK(rows: Row[], k: number, col: string, desc: boolean): Row[] {
  const heap: Row[] = [];

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

  for (const row of rows) {
    if (heap.length < k) { heap.push(row); siftUp(heap.length - 1); }
    else if (shouldReplace(row)) { heap[0] = row; siftDown(heap, 0); }
  }

  // Extract sorted
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

// --- Vector search ---

/** Cosine similarity vector search via WASM SIMD. */
function vectorSearch(
  columnData: Map<string, ArrayBuffer[]>,
  projectedColumns: ColumnMeta[],
  query: QueryDescriptor,
  wasmEngine: WasmEngine,
): Row[] {
  const vs = query.vectorSearch!;
  const embCol = projectedColumns.find(c => c.name === vs.column);
  if (!embCol || embCol.dtype !== "fixed_size_list") return [];

  const dim = embCol.listDimension ?? vs.queryVector.length;
  if (dim <= 0) return [];

  const embPages = columnData.get(vs.column);
  if (!embPages?.length) return [];

  // Flatten embedding pages
  let totalBytes = 0;
  for (const p of embPages) totalBytes += p.byteLength;
  const flat = new Float32Array(totalBytes / 4);
  let off = 0;
  for (const p of embPages) { const f = new Float32Array(p); flat.set(f, off); off += f.length; }
  const numRows = Math.floor(flat.length / dim);

  const embeddings: Float32Array[] = [];
  for (let r = 0; r < numRows; r++) embeddings.push(flat.subarray(r * dim, (r + 1) * dim));

  // Decode non-embedding columns
  const otherCols = decodeAllColumns(columnData, projectedColumns, vs.column, wasmEngine);

  wasmEngine.reset();
  const { indices, scores } = wasmEngine.vectorSearchBuffer(flat, numRows, dim, vs.queryVector, vs.topK);
  const rows: Row[] = [];
  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i];
    const row: Row = { _distance: 1 - scores[i], _score: scores[i] };
    for (const col of projectedColumns) {
      if (col.name === vs.column) row[col.name] = embeddings[idx] ?? null;
      else { const vals = otherCols.get(col.name); row[col.name] = vals ? (vals[idx] as Row[string]) : null; }
    }
    rows.push(row);
  }
  return rows;
}

// --- Filters ---

/** Coerce bigint↔number for cross-type comparisons (filter values are numbers, int64 columns decode as bigint). */
function coerceCompare(a: unknown, b: unknown): [unknown, unknown] {
  if (typeof a === "bigint" && typeof b === "number") return [a, BigInt(Math.trunc(b))];
  if (typeof a === "number" && typeof b === "bigint") return [BigInt(Math.trunc(a)), b];
  return [a, b];
}

export function matchesFilter(
  val: number | bigint | string | boolean | Float32Array | null,
  filter: QueryDescriptor["filters"][0],
): boolean {
  if (filter.op === "is_null") return val === null;
  if (filter.op === "is_not_null") return val !== null;
  if (val === null) return false;
  const t = filter.value;
  switch (filter.op) {
    case "eq":  { const [a, b] = coerceCompare(val, t); return a === b; }
    case "neq": { const [a, b] = coerceCompare(val, t); return a !== b; }
    case "gt":  { const [a, b] = coerceCompare(val, t); return (a as number | bigint | string) > (b as number | bigint | string); }
    case "gte": { const [a, b] = coerceCompare(val, t); return (a as number | bigint | string) >= (b as number | bigint | string); }
    case "lt":  { const [a, b] = coerceCompare(val, t); return (a as number | bigint | string) < (b as number | bigint | string); }
    case "lte": { const [a, b] = coerceCompare(val, t); return (a as number | bigint | string) <= (b as number | bigint | string); }
    case "in": {
      if (!Array.isArray(t)) return false;
      const set = getInSet(t);
      return set.has(val);
    }
    case "not_in": {
      if (!Array.isArray(t)) return false;
      const set = getInSet(t);
      return !set.has(val);
    }
    case "between": {
      if (!Array.isArray(t) || t.length !== 2) return false;
      const [, lo] = coerceCompare(val, t[0]);
      const [a, hi] = coerceCompare(val, t[1]);
      return (a as number | bigint | string) >= (lo as number | bigint | string) &&
             (a as number | bigint | string) <= (hi as number | bigint | string);
    }
    case "not_between": {
      if (!Array.isArray(t) || t.length !== 2) return false;
      const [, lo] = coerceCompare(val, t[0]);
      const [a, hi] = coerceCompare(val, t[1]);
      return (a as number | bigint | string) < (lo as number | bigint | string) ||
             (a as number | bigint | string) > (hi as number | bigint | string);
    }
    case "like": {
      if (typeof val !== "string" || typeof t !== "string") return false;
      return compileLikeRegex(t).test(val);
    }
    case "not_like": {
      if (typeof val !== "string" || typeof t !== "string") return false;
      return !compileLikeRegex(t).test(val);
    }
    default:    return true;
  }
}

/** Cache IN/NOT_IN value sets — O(1) lookup instead of O(m) per row. */
const inSetCache = new WeakMap<readonly (number | bigint | string)[], Set<number | bigint | string>>();

function getInSet(values: readonly (number | bigint | string)[]): Set<number | bigint | string> {
  let cached = inSetCache.get(values);
  if (cached) return cached;
  const set = new Set<number | bigint | string>(values);
  inSetCache.set(values, set);
  return set;
}

/** Cache compiled LIKE regexes — avoids re-compilation per row. */
const likeRegexCache = new Map<string, RegExp>();

function compileLikeRegex(pattern: string): RegExp {
  let cached = likeRegexCache.get(pattern);
  if (cached) return cached;
  // Escape regex metacharacters, then replace SQL wildcards
  const escaped = pattern.replace(/[.+?^${}()|[\]\\*]/g, "\\$&");
  const re = new RegExp("^" + escaped.replace(/%/g, ".*").replace(/_/g, ".") + "$", "i");
  likeRegexCache.set(pattern, re);
  if (likeRegexCache.size > 1000) likeRegexCache.clear(); // prevent unbounded growth
  return re;
}

export function bigIntReplacer(_key: string, value: unknown): unknown {
  return typeof value === "bigint" ? value.toString() : value;
}
