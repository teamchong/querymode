import type { ColumnMeta, PageInfo, Row } from "./types.js";
import type { QueryDescriptor } from "./client.js";
import type { WasmEngine } from "./wasm-engine.js";

const textDecoder = new TextDecoder();

/** Check if a page can be skipped via min/max stats. */
export function canSkipPage(page: PageInfo, filters: QueryDescriptor["filters"], columnName: string): boolean {
  for (const filter of filters) {
    if (filter.column !== columnName) continue;
    if (page.minValue === undefined || page.maxValue === undefined) continue;

    const { minValue: min, maxValue: max } = page;
    const val = filter.value;
    if (typeof val === "object") continue;

    switch (filter.op) {
      case "gt":  if (max <= val) return true; break;
      case "gte": if (max < val) return true; break;
      case "lt":  if (min >= val) return true; break;
      case "lte": if (min > val) return true; break;
      case "eq":  if (val < min || val > max) return true; break;
    }
  }
  return false;
}

/** Assemble rows from column page buffers. Uses WASM SIMD for vector search when available. */
export function assembleRows(
  columnData: Map<string, ArrayBuffer[]>,
  projectedColumns: ColumnMeta[],
  query: QueryDescriptor,
  wasmEngine?: WasmEngine | null,
): Row[] {
  if (query.vectorSearch) {
    return vectorSearch(columnData, projectedColumns, query, wasmEngine);
  }

  const decoded = decodeAllColumns(columnData, projectedColumns);
  const firstCol = decoded.values().next().value;
  if (!firstCol) return [];
  const rowCount = firstCol.length;

  const rows: Row[] = [];
  const earlyStop = (!query.sortColumn && query.limit) ? query.limit : rowCount;

  for (let i = 0; i < rowCount && rows.length < earlyStop; i++) {
    const row: Row = {};
    for (const col of projectedColumns) {
      const values = decoded.get(col.name);
      row[col.name] = values ? (values[i] as Row[string]) : null;
    }

    if (query.filters.every(f => {
      const v = row[f.column];
      return v !== null && matchesFilter(v, f);
    })) {
      rows.push(row);
    }
  }

  return applySortAndLimit(rows, query);
}

// --- Column decoding ---

type DecodedValue = number | bigint | string | boolean | Float32Array | null;

/** Decode all projected columns from their page buffers. */
function decodeAllColumns(
  columnData: Map<string, ArrayBuffer[]>,
  columns: ColumnMeta[],
  exclude?: string,
): Map<string, DecodedValue[]> {
  const result = new Map<string, DecodedValue[]>();
  for (const col of columns) {
    if (col.name === exclude) continue;
    const pages = columnData.get(col.name);
    if (!pages?.length) continue;

    if (col.dtype === "fixed_size_list") {
      result.set(col.name, decodeFixedSizeListPages(pages, col.listDimension ?? 0));
    } else if (col.dtype === "bool") {
      result.set(col.name, decodeBoolPages(pages, col.pages));
    } else {
      const values: DecodedValue[] = [];
      for (let i = 0; i < pages.length; i++) {
        const pi = col.pages[i];
        for (const v of decodePage(pages[i], col.dtype, pi?.nullCount ?? 0, pi?.rowCount ?? 0)) {
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

function decodeBoolPages(pages: ArrayBuffer[], pageInfos: PageInfo[]): boolean[] {
  const result: boolean[] = [];
  for (let i = 0; i < pages.length; i++) {
    const bytes = new Uint8Array(pages[i]);
    const rowCount = pageInfos[i]?.rowCount ?? (bytes.length * 8);
    let decoded = 0;
    for (let b = 0; b < bytes.length && decoded < rowCount; b++) {
      for (let bit = 0; bit < 8 && decoded < rowCount; bit++, decoded++) {
        result.push(((bytes[b] >> bit) & 1) === 1);
      }
    }
  }
  return result;
}

/** Decode a raw page buffer into typed values. Handles null bitmaps when nullCount > 0. */
export function decodePage(
  buf: ArrayBuffer,
  dtype: string,
  nullCount = 0,
  rowCount = 0,
): (number | bigint | string | null)[] {
  let nulls: Set<number> | null = null;
  if (nullCount > 0 && rowCount > 0) {
    const bitmapBytes = Math.ceil(rowCount / 8);
    const bytes = new Uint8Array(buf, 0, bitmapBytes);
    nulls = new Set<number>();
    let idx = 0;
    for (let b = 0; b < bitmapBytes && idx < rowCount; b++) {
      for (let bit = 0; bit < 8 && idx < rowCount; bit++, idx++) {
        if (((bytes[b] >> bit) & 1) === 0) nulls.add(idx);
      }
    }
    buf = buf.slice(bitmapBytes);
  }

  const view = new DataView(buf);
  const values: (number | bigint | string | null)[] = [];

  switch (dtype) {
    case "int8":    { const a = new Int8Array(buf); for (let i = 0; i < a.length; i++) values.push(a[i]); break; }
    case "uint8":   { const a = new Uint8Array(buf); for (let i = 0; i < a.length; i++) values.push(a[i]); break; }
    case "int16":   { const n = buf.byteLength >> 1; for (let i = 0; i < n; i++) values.push(view.getInt16(i * 2, true)); break; }
    case "uint16":  { const n = buf.byteLength >> 1; for (let i = 0; i < n; i++) values.push(view.getUint16(i * 2, true)); break; }
    case "int32":   { const n = buf.byteLength >> 2; for (let i = 0; i < n; i++) values.push(view.getInt32(i * 4, true)); break; }
    case "uint32":  { const n = buf.byteLength >> 2; for (let i = 0; i < n; i++) values.push(view.getUint32(i * 4, true)); break; }
    case "int64":   { const n = buf.byteLength >> 3; for (let i = 0; i < n; i++) values.push(view.getBigInt64(i * 8, true)); break; }
    case "uint64":  { const n = buf.byteLength >> 3; for (let i = 0; i < n; i++) values.push(view.getBigUint64(i * 8, true)); break; }
    case "float32": { const n = buf.byteLength >> 2; for (let i = 0; i < n; i++) values.push(view.getFloat32(i * 4, true)); break; }
    case "float64": { const n = buf.byteLength >> 3; for (let i = 0; i < n; i++) values.push(view.getFloat64(i * 8, true)); break; }
    case "float16": {
      const n = buf.byteLength >> 1;
      for (let i = 0; i < n; i++) {
        const h = view.getUint16(i * 2, true);
        const s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
        if (e === 0) values.push((s ? -1 : 1) * 2 ** -14 * (m / 1024));
        else if (e === 31) values.push(m ? NaN : s ? -Infinity : Infinity);
        else values.push((s ? -1 : 1) * 2 ** (e - 15) * (1 + m / 1024));
      }
      break;
    }
    case "utf8": {
      const bytes = new Uint8Array(buf);
      let pos = 0;
      while (pos + 4 <= bytes.length) {
        const len = view.getUint32(pos, true); pos += 4;
        if (pos + len > bytes.length) break;
        values.push(textDecoder.decode(bytes.subarray(pos, pos + len)));
        pos += len;
      }
      break;
    }
    case "binary": {
      let pos = 0;
      while (pos + 4 <= buf.byteLength) {
        const len = view.getUint32(pos, true); pos += 4;
        if (pos + len > buf.byteLength) break;
        const b = new Uint8Array(buf, pos, len);
        let hex = "";
        for (let i = 0; i < b.length; i++) hex += b[i].toString(16).padStart(2, "0");
        values.push(hex);
        pos += len;
      }
      break;
    }
  }

  if (nulls && nulls.size > 0) {
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
  if (!query.sortColumn || rows.length === 0) return rows;

  const col = query.sortColumn;
  const desc = query.sortDirection === "desc";

  if (query.limit && query.limit < rows.length) {
    return topK(rows, query.limit, col, desc);
  }

  const dir = desc ? -1 : 1;
  rows.sort((a, b) => {
    const av = a[col], bv = b[col];
    if (av === null && bv === null) return 0;
    if (av === null) return 1;
    if (bv === null) return -1;
    return av < bv ? -dir : av > bv ? dir : 0;
  });
  return query.limit ? rows.slice(0, query.limit) : rows;
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

  const siftDown = (i: number): void => {
    while (true) {
      let t = i;
      const l = 2 * i + 1, r = 2 * i + 2;
      if (l < heap.length && cmp(heap[l], heap[t]) > 0) t = l;
      if (r < heap.length && cmp(heap[r], heap[t]) > 0) t = r;
      if (t === i) break;
      [heap[i], heap[t]] = [heap[t], heap[i]];
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
    else if (shouldReplace(row)) { heap[0] = row; siftDown(0); }
  }

  // Extract sorted
  const result: Row[] = [];
  const copy = [...heap];
  while (copy.length > 0) {
    result.push(copy[0]);
    copy[0] = copy[copy.length - 1];
    copy.pop();
    if (copy.length > 0) {
      // inline siftDown on copy
      let i = 0;
      while (true) {
        let t = i;
        const l = 2 * i + 1, r = 2 * i + 2;
        if (l < copy.length && cmp(copy[l], copy[t]) > 0) t = l;
        if (r < copy.length && cmp(copy[r], copy[t]) > 0) t = r;
        if (t === i) break;
        [copy[i], copy[t]] = [copy[t], copy[i]];
        i = t;
      }
    }
  }
  result.reverse();
  return result;
}

// --- Vector search ---

/** Cosine similarity vector search. WASM SIMD fast path, TS fallback. */
function vectorSearch(
  columnData: Map<string, ArrayBuffer[]>,
  projectedColumns: ColumnMeta[],
  query: QueryDescriptor,
  wasmEngine?: WasmEngine | null,
): Row[] {
  const vs = query.vectorSearch!;
  const embCol = projectedColumns.find(c => c.name === vs.column);
  if (!embCol || embCol.dtype !== "fixed_size_list") return [];

  const dim = embCol.listDimension ?? vs.queryVector.length;
  if (dim === 0) return [];

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
  const otherCols = decodeAllColumns(columnData, projectedColumns, vs.column);

  // WASM SIMD path
  if (wasmEngine) {
    try {
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
    } catch { /* fall through to TS */ }
  }

  // TS cosine similarity fallback
  const qv = vs.queryVector;
  let qNorm = 0;
  for (let d = 0; d < dim; d++) qNorm += qv[d] * qv[d];
  qNorm = Math.sqrt(qNorm);

  // Top-K via max-heap on cosine distance
  const heap: { dist: number; idx: number }[] = [];
  for (let i = 0; i < numRows; i++) {
    const emb = embeddings[i];
    let dot = 0, eNorm = 0;
    for (let d = 0; d < dim; d++) { dot += emb[d] * qv[d]; eNorm += emb[d] * emb[d]; }
    const dist = 1 - ((qNorm > 0 && eNorm > 0) ? dot / (qNorm * Math.sqrt(eNorm)) : 0);

    if (heap.length < vs.topK) {
      heap.push({ dist, idx: i });
      let j = heap.length - 1;
      while (j > 0) { const p = (j - 1) >> 1; if (heap[j].dist > heap[p].dist) { [heap[j], heap[p]] = [heap[p], heap[j]]; j = p; } else break; }
    } else if (dist < heap[0].dist) {
      heap[0] = { dist, idx: i };
      let j = 0;
      while (true) {
        let t = j; const l = 2 * j + 1, r = 2 * j + 2;
        if (l < heap.length && heap[l].dist > heap[t].dist) t = l;
        if (r < heap.length && heap[r].dist > heap[t].dist) t = r;
        if (t === j) break;
        [heap[j], heap[t]] = [heap[t], heap[j]]; j = t;
      }
    }
  }

  heap.sort((a, b) => a.dist - b.dist);
  return heap.map(({ dist, idx }) => {
    const row: Row = { _distance: dist, _score: 1 - dist };
    for (const col of projectedColumns) {
      if (col.name === vs.column) row[col.name] = embeddings[idx] ?? null;
      else { const vals = otherCols.get(col.name); row[col.name] = vals ? (vals[idx] as Row[string]) : null; }
    }
    return row;
  });
}

// --- Filters ---

export function matchesFilter(
  val: number | bigint | string | boolean | Float32Array | null,
  filter: QueryDescriptor["filters"][0],
): boolean {
  if (val === null) return false;
  const t = filter.value;
  switch (filter.op) {
    case "eq":  return val === t;
    case "neq": return val !== t;
    case "gt":  return val > (t as number | bigint | string);
    case "gte": return val >= (t as number | bigint | string);
    case "lt":  return val < (t as number | bigint | string);
    case "lte": return val <= (t as number | bigint | string);
    case "in":  return Array.isArray(t) && t.includes(val as number | bigint | string);
    default:    return true;
  }
}

export function bigIntReplacer(_key: string, value: unknown): unknown {
  return typeof value === "bigint" ? value.toString() : value;
}
