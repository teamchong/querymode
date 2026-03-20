import type { ColumnMeta, FilterOp, PageInfo, Row, DataType } from "./types.js";
import { rowComparator } from "./types.js";
import type { QueryDescriptor } from "./client.js";
import type { WasmEngine } from "./wasm-engine.js";
import { decodeParquetColumnChunk, decodePlainValues } from "./parquet-decode.js";
import { decodeLanceV2Utf8 } from "./lance-v2.js";

/** Check if a page can be skipped via min/max stats. */
export function canSkipPage(page: PageInfo, filters: QueryDescriptor["filters"], columnName: string): boolean {
  for (const filter of filters) {
    if (filter.column !== columnName) continue;
    // is_null: skip when page has no nulls; is_not_null: skip when page is all nulls
    if (filter.op === "is_null") { if (page.nullCount === 0) return true; continue; }
    if (filter.op === "is_not_null") { if (page.nullCount === page.rowCount) return true; continue; }
    if (page.minValue === undefined || page.maxValue === undefined) continue;

    let { minValue: min, maxValue: max } = page;
    let val = filter.value;
    if (typeof val === "object" && !Array.isArray(val)) continue;

    // Coerce bigint↔number for cross-type comparisons (skip non-finite numbers — BigInt(NaN) throws)
    if (typeof min === "bigint" && typeof val === "number") { if (!Number.isFinite(val)) continue; val = BigInt(Math.trunc(val as number)); }
    else if (typeof min === "number" && typeof val === "bigint") { if (!Number.isFinite(min) || !Number.isFinite(max as number)) continue; min = BigInt(Math.trunc(min)); max = BigInt(Math.trunc(max as number)); }

    switch (filter.op) {
      case "gt":  if (max <= val) return true; break;
      case "gte": if (max < val) return true; break;
      case "lt":  if (min >= val) return true; break;
      case "lte": if (min > val) return true; break;
      case "eq":  if (val < min || val > max) return true; break;
      // neq: skip only when entire page is a single value equal to the filter value
      case "neq": if (min === max && min === val) return true; break;
      // in: skip when all IN values fall outside the page's [min, max] range
      case "in": {
        if (!Array.isArray(filter.value)) break;
        if (filter.value.every(v => {
          let cv = v;
          let cmin = min, cmax = max;
          if (typeof cmin === "bigint" && typeof cv === "number") { if (!Number.isFinite(cv)) return false; cv = BigInt(Math.trunc(cv)); }
          else if (typeof cmin === "number" && typeof cv === "bigint") { if (!Number.isFinite(cmin as number) || !Number.isFinite(cmax as number)) return false; cmin = BigInt(Math.trunc(cmin as number)); cmax = BigInt(Math.trunc(cmax as number)); }
          return cv < cmin || cv > cmax;
        })) return true;
        break;
      }
      // not_in: skip when entire page is a single value that appears in the NOT IN list
      case "not_in": {
        if (!Array.isArray(filter.value) || min !== max) break;
        if (filter.value.some(v => {
          let cv = v;
          let cmin = min;
          if (typeof cmin === "bigint" && typeof cv === "number") { if (!Number.isFinite(cv)) return false; cv = BigInt(Math.trunc(cv)); }
          else if (typeof cmin === "number" && typeof cv === "bigint") { if (!Number.isFinite(cmin as number)) return false; cmin = BigInt(Math.trunc(cmin as number)); }
          return cv === cmin;
        })) return true;
        break;
      }
      case "between": {
        if (!Array.isArray(filter.value) || filter.value.length !== 2) break;
        let lo = filter.value[0];
        let hi = filter.value[1];
        if (typeof min === "bigint" && typeof lo === "number") { if (!Number.isFinite(lo) || !Number.isFinite(hi as number)) break; lo = BigInt(Math.trunc(lo)); hi = BigInt(Math.trunc(hi as number)); }
        else if (typeof min === "number" && typeof lo === "bigint") { if (!Number.isFinite(min) || !Number.isFinite(max as number)) break; min = BigInt(Math.trunc(min)); max = BigInt(Math.trunc(max as number)); }
        if (max < lo || min > hi) return true;
        break;
      }
      case "not_between": {
        if (!Array.isArray(filter.value) || filter.value.length !== 2) break;
        let lo = filter.value[0];
        let hi = filter.value[1];
        if (typeof min === "bigint" && typeof lo === "number") { if (!Number.isFinite(lo) || !Number.isFinite(hi as number)) break; lo = BigInt(Math.trunc(lo)); hi = BigInt(Math.trunc(hi as number)); }
        else if (typeof min === "number" && typeof lo === "bigint") { if (!Number.isFinite(min) || !Number.isFinite(max as number)) break; min = BigInt(Math.trunc(min)); max = BigInt(Math.trunc(max as number)); }
        // Skip if all values are within [lo, hi] — NOT BETWEEN would exclude them all
        if (min >= lo && max <= hi) return true;
        break;
      }
      // like: skip when the fixed prefix doesn't overlap the page's string range
      case "like": {
        if (typeof filter.value !== "string" || typeof min !== "string") break;
        const prefix = extractLikePrefix(filter.value);
        if (!prefix) break;
        // Page max < prefix → no match possible
        if ((max as string) < prefix) { return true; }
        // Page min >= prefix successor → no match possible (guard U+FFFF overflow)
        const lastCode = prefix.charCodeAt(prefix.length - 1);
        if (lastCode < 0xffff) {
          const prefixEnd = prefix.slice(0, -1) + String.fromCharCode(lastCode + 1);
          if ((min as string) >= prefixEnd) return true;
        }
        break;
      }
      // not_like: skip uniform pages where the single value matches the pattern
      case "not_like": {
        if (typeof filter.value !== "string" || typeof min !== "string" || min !== max) break;
        if (compileLikeRegex(filter.value).test(min)) return true;
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
  filterGroups?: FilterOp[][],
): boolean {
  if (filters.length === 0 && (!filterGroups || filterGroups.length === 0)) return false;

  // Compute per-column min/max/nullCount/rowCount across all pages
  const colStats = new Map<string, { min?: number | bigint | string; max?: number | bigint | string; nullCount: number; rowCount: number }>();
  for (const col of meta.columns) {
    let colMin: number | bigint | string | undefined;
    let colMax: number | bigint | string | undefined;
    let hasStats = true;
    let totalNulls = 0;
    let totalRows = 0;
    for (const page of col.pages) {
      totalNulls += page.nullCount;
      totalRows += page.rowCount;
      if (hasStats) {
        if (page.minValue === undefined || page.maxValue === undefined) {
          hasStats = false;
        } else {
          if (colMin === undefined || page.minValue < colMin) colMin = page.minValue;
          if (colMax === undefined || page.maxValue > colMax) colMax = page.maxValue;
        }
      }
    }
    colStats.set(col.name, {
      min: hasStats ? colMin : undefined,
      max: hasStats ? colMax : undefined,
      nullCount: totalNulls,
      rowCount: totalRows,
    });
  }

  const synthPage = (s: { min?: number | bigint | string; max?: number | bigint | string; nullCount: number; rowCount: number }): PageInfo =>
    ({ byteOffset: 0n, byteLength: 0, rowCount: s.rowCount, nullCount: s.nullCount, minValue: s.min, maxValue: s.max });

  const filterSkips = (f: FilterOp): boolean => {
    const stats = colStats.get(f.column);
    return stats ? canSkipPage(synthPage(stats), [f], f.column) : false;
  };

  // Build a synthetic "fragment page" with the aggregated stats and reuse canSkipPage
  for (const f of filters) if (filterSkips(f)) return true;

  // OR groups: skip only if ALL groups eliminate the fragment
  if (filterGroups && filterGroups.length > 0) {
    if (filterGroups.every(group => group.some(f => filterSkips(f)))) return true;
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

    if (!rowPassesFilters(row, query.filters, query.filterGroups)) continue;

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
    const numRows = Math.floor((pageBuf.byteLength >> 2) / dim);
    // One flat allocation + subarray views — O(1) allocations per page instead of O(numRows).
    const flat = new Float32Array(numRows * dim);
    const view = new DataView(pageBuf);
    for (let i = 0; i < numRows * dim; i++) flat[i] = view.getFloat32(i * 4, true);
    for (let r = 0; r < numRows; r++) result.push(flat.subarray(r * dim, (r + 1) * dim));
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
  let nullBitmap: Uint8Array | null = null;
  let nulls: { has(i: number): boolean; size: number } | null = null;
  if (nullCount > 0 && rowCount > 0) {
    const bitmapBytes = Math.ceil(rowCount / 8);
    if (buf.byteLength < bitmapBytes) return [];
    nullBitmap = new Uint8Array(buf, 0, bitmapBytes);
    nulls = {
      has(i: number): boolean { return (nullBitmap![i >> 3] & (1 << (i & 7))) === 0; },
      get size() { return nullCount; },
    };
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
    let v2Valid = v2Strings.length === decodeCount;
    if (v2Valid) {
      for (let i = 0; i < v2Strings.length; i++) {
        if (typeof v2Strings[i] !== "string" || v2Strings[i].length >= buf.byteLength) { v2Valid = false; break; }
      }
    }
    if (v2Valid) {
      if (nulls && nulls.size > 0) {
        if (isLanceV2Nullable) {
          // Lance v2: all slots present, null-mask in place
          for (let i = 0; i < v2Strings.length; i++) if (nulls.has(i)) (v2Strings as (string | null)[])[i] = null;
          return v2Strings as (string | null)[];
        }
        // Parquet-style: packed non-null values, interleave with nulls
        const withNulls = new Array<string | null>(rowCount);
        let vi = 0;
        for (let i = 0; i < rowCount; i++) {
          withNulls[i] = nulls.has(i) ? null : (vi < v2Strings.length ? v2Strings[vi++] : null);
        }
        return withNulls;
      }
      return v2Strings;
    }
  }

  const values = decodePlainValues(bytes, dtype as DataType, numValues) as (number | bigint | string | null)[];

  if (nulls && nulls.size > 0) {
    if (isLanceV2Nullable) {
      // Lance v2: all row slots present, null-mask in place
      for (let i = 0; i < values.length; i++) if (nulls.has(i)) values[i] = null;
      return values;
    }
    // Parquet-style: packed non-null values, interleave with nulls
    const withNulls = new Array<number | bigint | string | null>(rowCount);
    let vi = 0;
    for (let i = 0; i < rowCount; i++) {
      withNulls[i] = nulls.has(i) ? null : (vi < values.length ? values[vi++] : null);
    }
    return withNulls;
  }

  return values;
}

// --- Sort / Top-K ---

export function applySortAndLimit(rows: Row[], query: QueryDescriptor): Row[] {
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

  rows.sort(rowComparator(col, desc));
  const start = offset;
  const end = limit !== undefined ? offset + limit : undefined;
  return rows.slice(start, end);
}

/** Top-K via max-heap (asc) or min-heap (desc). O(N log K) time, O(K) memory. */
function topK(rows: Row[], k: number, col: string, desc: boolean): Row[] {
  const heap: Row[] = [];

  const cmp = (a: Row, b: Row): number => {
    const av = a[col], bv = b[col];
    // Nulls-last: null is "greatest" so it sits at the heap root and gets replaced first
    if ((av === null || av === undefined) && (bv === null || bv === undefined)) return 0;
    if (av === null || av === undefined) return 1;
    if (bv === null || bv === undefined) return -1;
    if (typeof av === "number" && isNaN(av)) return typeof bv === "number" && isNaN(bv) ? 0 : 1;
    if (typeof bv === "number" && isNaN(bv)) return -1;
    const c = av < bv ? -1 : av > bv ? 1 : 0;
    return desc ? -c : c;
  };

  const shouldReplace = (row: Row): boolean => {
    const nv = row[col], rv = heap[0][col];
    if (nv === null || nv === undefined) return false;
    if (typeof nv === "number" && isNaN(nv)) return false;
    if (rv === null || rv === undefined) return true;
    if (typeof rv === "number" && isNaN(rv)) return true;
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

/** Coerce bigint↔number for cross-type comparisons (filter values are numbers, int64 columns decode as bigint).
 *  Returns coerced values via out parameter to avoid tuple allocation on hot path. */
const _cmp: [unknown, unknown] = [null, null];
function coerceCompare(a: unknown, b: unknown): [unknown, unknown] {
  if (typeof a === "bigint" && typeof b === "number") { _cmp[0] = a; _cmp[1] = Number.isFinite(b) ? BigInt(Math.trunc(b)) : b; }
  else if (typeof a === "number" && typeof b === "bigint") { _cmp[0] = Number.isFinite(a) ? BigInt(Math.trunc(a)) : a; _cmp[1] = b; }
  else { _cmp[0] = a; _cmp[1] = b; }
  return _cmp;
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
      return getInSet(t).has(val as number | bigint | string);
    }
    case "not_in": {
      if (!Array.isArray(t)) return false;
      return !getInSet(t).has(val as number | bigint | string);
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
  // Add cross-type equivalents so set.has() works for both bigint and number
  // representations without unsafe Number(bigint) coercion that loses precision.
  for (const v of values) {
    if (typeof v === "number" && Number.isFinite(v) && Number.isInteger(v)) {
      set.add(BigInt(v));
    } else if (typeof v === "bigint") {
      // Only add Number equivalent if the bigint fits safely
      if (v >= -9007199254740991n && v <= 9007199254740991n) {
        set.add(Number(v));
      }
    }
  }
  inSetCache.set(values, set);
  return set;
}

/** Test whether a row passes AND filters + OR filter groups. */
export function rowPassesFilters(row: Row, filters: FilterOp[], filterGroups?: FilterOp[][]): boolean {
  for (const f of filters) {
    const v = row[f.column];
    if (f.op === "is_null") { if (v !== null && v !== undefined) return false; continue; }
    if (f.op === "is_not_null") { if (v === null || v === undefined) return false; continue; }
    if (v === null || v === undefined) return false;
    if (!matchesFilter(v, f)) return false;
  }
  if (filterGroups && filterGroups.length > 0) {
    return filterGroups.some(group =>
      group.every(f => {
        const v = row[f.column];
        if (f.op === "is_null") return v === null || v === undefined;
        if (f.op === "is_not_null") return v !== null && v !== undefined;
        if (v === null || v === undefined) return false;
        return matchesFilter(v, f);
      }),
    );
  }
  return true;
}

/** Extract the fixed prefix from a LIKE pattern (before the first wildcard). Returns null if no useful prefix. */
function extractLikePrefix(pattern: string): string | null {
  let prefix = "";
  for (let i = 0; i < pattern.length; i++) {
    const ch = pattern[i];
    if (ch === "%" || ch === "_") break;
    if (ch === "\\" && i + 1 < pattern.length) { prefix += pattern[++i]; continue; }
    prefix += ch;
  }
  return prefix.length > 0 ? prefix : null;
}

/** Cache compiled LIKE regexes — avoids re-compilation per row. */
const likeRegexCache = new Map<string, RegExp>();

export function compileLikeRegex(pattern: string): RegExp {
  let cached = likeRegexCache.get(pattern);
  if (cached) return cached;
  // Walk character-by-character to handle SQL escape sequences (\%, \_)
  let reStr = "^";
  for (let i = 0; i < pattern.length; i++) {
    const ch = pattern[i];
    if (ch === "\\" && i + 1 < pattern.length) {
      // SQL escape: \% → literal %, \_ → literal _, \\ → literal \
      const next = pattern[++i];
      reStr += next.replace(/[.+?^${}()|[\]\\*]/g, "\\$&");
    } else if (ch === "%") {
      reStr += ".*";
    } else if (ch === "_") {
      reStr += ".";
    } else {
      reStr += ch.replace(/[.+?^${}()|[\]\\*]/g, "\\$&");
    }
  }
  reStr += "$";
  const re = new RegExp(reStr, "s");
  likeRegexCache.set(pattern, re);
  if (likeRegexCache.size > 1000) likeRegexCache.clear(); // prevent unbounded growth
  return re;
}

export function bigIntReplacer(_key: string, value: unknown): unknown {
  return typeof value === "bigint" ? value.toString() : value;
}
