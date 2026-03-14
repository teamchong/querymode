/**
 * Zero-copy columnar data exchange for the query pipeline.
 *
 * Converts WASM query results directly to QMCB (QueryMode Columnar Binary)
 * format without creating intermediate Row[] JS objects. This eliminates
 * the biggest serialization bottleneck: WASM → Row[] → JSON over RPC.
 *
 * Data flow:
 *   WASM executeSql → result in WASM memory (already columnar)
 *   → wasmResultToQMCB() reads WASM memory, writes QMCB (single memcpy for numerics)
 *   → ArrayBuffer transferred over Worker RPC (structured clone, not JSON)
 *   → decodeColumnarBatch() at receiver (views into buffer)
 *   → columnarBatchToRows() only at final response boundary
 */

import type { Row } from "./types.js";

const textDecoder = new TextDecoder();
const textEncoder = new TextEncoder();

const QMCB_MAGIC = 0x42434D51; // "QMCB" little-endian

// QMCB dtype tags (extends r2-spill.ts)
export const DTYPE_F64 = 0;
export const DTYPE_I64 = 1;
export const DTYPE_UTF8 = 2;
export const DTYPE_BOOL = 3;
export const DTYPE_F32VEC = 4;
export const DTYPE_NULL = 5;
export const DTYPE_I32 = 6;
export const DTYPE_F32 = 7;

// WASM ColumnType (from sql_executor.zig)
const WASM_INT64 = 0;
const WASM_FLOAT64 = 1;
const WASM_STRING = 2;
const WASM_BOOL = 3;
const WASM_INT32 = 4;
const WASM_FLOAT32 = 5;
const WASM_VECTOR = 6;

/** A single column in a columnar batch. */
export interface ColumnarColumn {
  name: string;
  dtype: number;
  data: ArrayBuffer;
  rowCount: number;
  /** String columns: byte offsets (rowCount+1 entries). */
  offsets?: Uint32Array;
  /** Vector columns: dimension. */
  vectorDim?: number;
  /** Null bitmap: bit set = null. */
  nullBitmap?: Uint8Array;
}

/** A batch of columnar data — the unit of zero-copy transfer. */
export interface ColumnarBatch {
  columns: ColumnarColumn[];
  rowCount: number;
}

function wasmTypeToDtype(t: number): number {
  switch (t) {
    case WASM_INT64: return DTYPE_I64;
    case WASM_FLOAT64: return DTYPE_F64;
    case WASM_STRING: return DTYPE_UTF8;
    case WASM_BOOL: return DTYPE_BOOL;
    case WASM_INT32: return DTYPE_I32;
    case WASM_FLOAT32: return DTYPE_F32;
    case WASM_VECTOR: return DTYPE_F32VEC;
    default: return DTYPE_NULL;
  }
}

function bytesPerElement(dtype: number): number {
  switch (dtype) {
    case DTYPE_F64: case DTYPE_I64: return 8;
    case DTYPE_I32: case DTYPE_F32: return 4;
    default: return 0;
  }
}

// ============================================================================
// WASM result → QMCB (single-pass, zero intermediate objects)
// ============================================================================

/**
 * Convert WASM query result directly to QMCB binary format.
 *
 * Reads the columnar result in WASM linear memory and writes QMCB
 * in a single pass. For numeric columns this is a memcpy. For strings
 * it reorganizes inline [len][bytes] into offset-based layout.
 *
 * MUST be called before resetHeap()/resetResult() — reads WASM memory directly.
 */
export function wasmResultToQMCB(
  memoryBuffer: ArrayBuffer,
  ptr: number,
  size: number,
): ArrayBuffer | null {
  if (size < 28) return null;

  const view = new DataView(memoryBuffer, ptr, size);
  const buf = new Uint8Array(memoryBuffer, ptr, size);

  const numColumns = view.getUint32(4, true);
  const numRows = view.getUint32(8, true);
  const dataStart = view.getUint32(16, true);
  if (!numColumns || !numRows) return null;

  // Parse column descriptors
  interface ColDesc { name: string; nameBytes: Uint8Array; wasmType: number; dtype: number }
  const cols: ColDesc[] = [];
  let pos = 28;
  for (let c = 0; c < numColumns; c++) {
    const nameLen = buf[pos++];
    const name = textDecoder.decode(buf.subarray(pos, pos + nameLen));
    pos += nameLen;
    const wasmType = buf[pos++];
    cols.push({ name, nameBytes: textEncoder.encode(name), wasmType, dtype: wasmTypeToDtype(wasmType) });
  }

  // Allocate output — QMCB is at most WASM result size + header overhead.
  // String columns add (numRows+1)*4 offsets but remove numRows*4 inline lengths = net +4 bytes.
  // Bool columns shrink (1 byte/row → 1 bit/row). Safe upper bound: size + 4096.
  const out = new ArrayBuffer(size + 4096);
  const outView = new DataView(out);
  const outBuf = new Uint8Array(out);
  let wp = 0;

  // QMCB header
  outView.setUint32(wp, QMCB_MAGIC, true); wp += 4;
  outView.setUint32(wp, numRows, true); wp += 4;
  outView.setUint16(wp, numColumns, true); wp += 2;

  // Column descriptors
  for (const col of cols) {
    outView.setUint16(wp, col.nameBytes.length, true); wp += 2;
    outBuf.set(col.nameBytes, wp); wp += col.nameBytes.length;
    outBuf[wp++] = col.dtype;
    outBuf[wp++] = 0; // hasNulls = false (WASM result format has no null bitmap)
  }

  // Column data — single pass over WASM result
  let dp = dataStart;

  for (const col of cols) {
    switch (col.wasmType) {
      case WASM_INT64:
      case WASM_FLOAT64: {
        const sz = numRows * 8;
        if (dp + sz > size) break;
        outBuf.set(new Uint8Array(memoryBuffer, ptr + dp, sz), wp);
        wp += sz;
        dp += sz;
        break;
      }

      case WASM_INT32:
      case WASM_FLOAT32: {
        const sz = numRows * 4;
        if (dp + sz > size) break;
        outBuf.set(new Uint8Array(memoryBuffer, ptr + dp, sz), wp);
        wp += sz;
        dp += sz;
        break;
      }

      case WASM_BOOL: {
        if (dp + numRows > size) break;
        const boolData = new Uint8Array(memoryBuffer, ptr + dp, numRows);
        const packedLen = Math.ceil(numRows / 8);
        // outBuf is zero-initialized (new ArrayBuffer), OR is safe
        for (let i = 0; i < numRows; i++) {
          if (boolData[i]) outBuf[wp + (i >> 3)] |= 1 << (i & 7);
        }
        wp += packedLen;
        dp += numRows;
        break;
      }

      case WASM_STRING: {
        // WASM: per row [u32 len][len bytes]
        // QMCB: [u32 totalLen][(n+1)*u32 offsets][bytes contiguous]
        //
        // Write totalLen at the end (fill after scanning all rows).
        // Reserve space for totalLen + offsets, then write string data inline.
        const totalLenPos = wp;
        wp += 4;

        const offsetsPos = wp;
        wp += (numRows + 1) * 4;

        const dataPos = wp;
        let strOffset = 0;

        for (let r = 0; r < numRows; r++) {
          if (dp + 4 > size) break;
          outView.setUint32(offsetsPos + r * 4, strOffset, true);
          const len = view.getUint32(dp, true);
          dp += 4;
          if (len > 0 && dp + len <= size) {
            outBuf.set(new Uint8Array(memoryBuffer, ptr + dp, len), dataPos + strOffset);
          }
          strOffset += len;
          dp += len;
        }
        outView.setUint32(offsetsPos + numRows * 4, strOffset, true);
        outView.setUint32(totalLenPos, strOffset, true);
        wp = dataPos + strOffset;
        break;
      }

      default: {
        // Unknown column type — skip 8 bytes per row in WASM result
        dp += numRows * 8;
        break;
      }
    }
  }

  // Return trimmed buffer
  return out.slice(0, wp);
}

// ============================================================================
// QMCB decode
// ============================================================================

/**
 * Decode QMCB binary into a ColumnarBatch.
 * Column data is sliced from the buffer (owned, properly aligned for typed arrays).
 */
export function decodeColumnarBatch(qmcb: ArrayBuffer): ColumnarBatch | null {
  if (qmcb.byteLength < 10) return null;

  const view = new DataView(qmcb);
  const buf = new Uint8Array(qmcb);

  if (view.getUint32(0, true) !== QMCB_MAGIC) return null;

  const rowCount = view.getUint32(4, true);
  const colCount = view.getUint16(8, true);

  interface Desc { name: string; dtype: number; hasNulls: boolean }
  const descs: Desc[] = [];
  let pos = 10;

  for (let c = 0; c < colCount; c++) {
    const nameLen = view.getUint16(pos, true); pos += 2;
    const name = textDecoder.decode(buf.subarray(pos, pos + nameLen)); pos += nameLen;
    const dtype = buf[pos++];
    const hasNulls = buf[pos++] !== 0;
    descs.push({ name, dtype, hasNulls });
  }

  const columns: ColumnarColumn[] = [];

  for (const desc of descs) {
    let nullBitmap: Uint8Array | undefined;
    if (desc.hasNulls) {
      const nbLen = Math.ceil(rowCount / 8);
      nullBitmap = new Uint8Array(qmcb.slice(pos, pos + nbLen));
      pos += nbLen;
    }

    switch (desc.dtype) {
      case DTYPE_F64:
      case DTYPE_I64: {
        const sz = rowCount * 8;
        columns.push({ name: desc.name, dtype: desc.dtype, rowCount, data: qmcb.slice(pos, pos + sz), nullBitmap });
        pos += sz;
        break;
      }
      case DTYPE_I32:
      case DTYPE_F32: {
        const sz = rowCount * 4;
        columns.push({ name: desc.name, dtype: desc.dtype, rowCount, data: qmcb.slice(pos, pos + sz), nullBitmap });
        pos += sz;
        break;
      }
      case DTYPE_UTF8: {
        const totalLen = view.getUint32(pos, true); pos += 4;
        const offsets = new Uint32Array(qmcb.slice(pos, pos + (rowCount + 1) * 4));
        pos += (rowCount + 1) * 4;
        const data = qmcb.slice(pos, pos + totalLen);
        pos += totalLen;
        columns.push({ name: desc.name, dtype: desc.dtype, rowCount, data, offsets, nullBitmap });
        break;
      }
      case DTYPE_BOOL: {
        const sz = Math.ceil(rowCount / 8);
        columns.push({ name: desc.name, dtype: desc.dtype, rowCount, data: qmcb.slice(pos, pos + sz), nullBitmap });
        pos += sz;
        break;
      }
      case DTYPE_F32VEC: {
        const dim = view.getUint32(pos, true); pos += 4;
        const sz = rowCount * dim * 4;
        columns.push({ name: desc.name, dtype: desc.dtype, rowCount, data: qmcb.slice(pos, pos + sz), vectorDim: dim, nullBitmap });
        pos += sz;
        break;
      }
      default: {
        columns.push({ name: desc.name, dtype: desc.dtype, rowCount, data: new ArrayBuffer(0), nullBitmap });
        break;
      }
    }
  }

  return { columns, rowCount };
}

// ============================================================================
// QMCB encode (from ColumnarBatch)
// ============================================================================

/** Encode a ColumnarBatch into QMCB binary format. */
export function encodeColumnarBatch(batch: ColumnarBatch): ArrayBuffer {
  const { columns, rowCount } = batch;

  const nameBytesList: Uint8Array[] = [];
  let size = 10; // header: magic(4) + rowCount(4) + colCount(2)
  for (const col of columns) {
    const nb = textEncoder.encode(col.name);
    nameBytesList.push(nb);
    size += 2 + nb.length + 1 + 1; // nameLen(2) + name + dtype(1) + hasNulls(1)
  }

  for (const col of columns) {
    if (col.nullBitmap) size += Math.ceil(rowCount / 8);
    switch (col.dtype) {
      case DTYPE_F64: case DTYPE_I64: size += rowCount * 8; break;
      case DTYPE_I32: case DTYPE_F32: size += rowCount * 4; break;
      case DTYPE_BOOL: size += Math.ceil(rowCount / 8); break;
      case DTYPE_UTF8: {
        const totalLen = col.offsets ? col.offsets[rowCount] : col.data.byteLength;
        size += 4 + (rowCount + 1) * 4 + totalLen;
        break;
      }
      case DTYPE_F32VEC: size += 4 + rowCount * (col.vectorDim || 0) * 4; break;
      default: break;
    }
  }

  const out = new ArrayBuffer(size);
  const outView = new DataView(out);
  const outBuf = new Uint8Array(out);
  let wp = 0;

  outView.setUint32(wp, QMCB_MAGIC, true); wp += 4;
  outView.setUint32(wp, rowCount, true); wp += 4;
  outView.setUint16(wp, columns.length, true); wp += 2;

  for (let ci = 0; ci < columns.length; ci++) {
    const nb = nameBytesList[ci];
    outView.setUint16(wp, nb.length, true); wp += 2;
    outBuf.set(nb, wp); wp += nb.length;
    outBuf[wp++] = columns[ci].dtype;
    outBuf[wp++] = columns[ci].nullBitmap ? 1 : 0;
  }

  for (const col of columns) {
    if (col.nullBitmap) {
      const nbLen = Math.ceil(rowCount / 8);
      outBuf.set(col.nullBitmap.subarray(0, nbLen), wp);
      wp += nbLen;
    }
    switch (col.dtype) {
      case DTYPE_F64: case DTYPE_I64: {
        outBuf.set(new Uint8Array(col.data, 0, rowCount * 8), wp);
        wp += rowCount * 8;
        break;
      }
      case DTYPE_I32: case DTYPE_F32: {
        outBuf.set(new Uint8Array(col.data, 0, rowCount * 4), wp);
        wp += rowCount * 4;
        break;
      }
      case DTYPE_BOOL: {
        const sz = Math.ceil(rowCount / 8);
        outBuf.set(new Uint8Array(col.data, 0, sz), wp);
        wp += sz;
        break;
      }
      case DTYPE_UTF8: {
        const offsets = col.offsets!;
        const totalLen = offsets[rowCount];
        outView.setUint32(wp, totalLen, true); wp += 4;
        for (let r = 0; r <= rowCount; r++) {
          outView.setUint32(wp, offsets[r], true); wp += 4;
        }
        outBuf.set(new Uint8Array(col.data, 0, totalLen), wp);
        wp += totalLen;
        break;
      }
      case DTYPE_F32VEC: {
        const dim = col.vectorDim || 0;
        outView.setUint32(wp, dim, true); wp += 4;
        outBuf.set(new Uint8Array(col.data, 0, rowCount * dim * 4), wp);
        wp += rowCount * dim * 4;
        break;
      }
      default: break;
    }
  }

  return out;
}

// ============================================================================
// Concat QMCB batches
// ============================================================================

/** Concatenate multiple QMCB batches with the same schema into one. */
export function concatQMCBBatches(batches: ArrayBuffer[]): ArrayBuffer | null {
  if (batches.length === 0) return null;
  if (batches.length === 1) return batches[0];

  const decoded = batches.map(b => decodeColumnarBatch(b)).filter((b): b is ColumnarBatch => b !== null);
  if (decoded.length === 0) return null;

  const totalRows = decoded.reduce((s, b) => s + b.rowCount, 0);
  const numCols = decoded[0].columns.length;

  const columns: ColumnarColumn[] = [];

  for (let ci = 0; ci < numCols; ci++) {
    const dtype = decoded[0].columns[ci].dtype;
    const name = decoded[0].columns[ci].name;
    const bpe = bytesPerElement(dtype);

    if (bpe > 0) {
      // Fixed-width numeric: memcpy concat
      const buf = new ArrayBuffer(totalRows * bpe);
      const out = new Uint8Array(buf);
      let offset = 0;
      for (const batch of decoded) {
        const src = new Uint8Array(batch.columns[ci].data);
        out.set(src, offset);
        offset += src.length;
      }
      columns.push({ name, dtype, data: buf, rowCount: totalRows });
    } else if (dtype === DTYPE_BOOL) {
      const buf = new ArrayBuffer(Math.ceil(totalRows / 8));
      const out = new Uint8Array(buf);
      let row = 0;
      for (const batch of decoded) {
        const src = new Uint8Array(batch.columns[ci].data);
        for (let r = 0; r < batch.rowCount; r++) {
          if (src[r >> 3] & (1 << (r & 7))) out[row >> 3] |= 1 << (row & 7);
          row++;
        }
      }
      columns.push({ name, dtype, data: buf, rowCount: totalRows });
    } else if (dtype === DTYPE_UTF8) {
      let totalStrLen = 0;
      for (const batch of decoded) {
        const col = batch.columns[ci];
        totalStrLen += col.offsets ? col.offsets[batch.rowCount] : col.data.byteLength;
      }

      const offsets = new Uint32Array(totalRows + 1);
      const strBuf = new Uint8Array(totalStrLen);
      let strOffset = 0;
      let row = 0;

      for (const batch of decoded) {
        const col = batch.columns[ci];
        const srcOffsets = col.offsets!;
        const srcData = new Uint8Array(col.data);
        for (let r = 0; r < batch.rowCount; r++) {
          offsets[row] = strOffset;
          const start = srcOffsets[r];
          const end = srcOffsets[r + 1];
          if (end > start) {
            strBuf.set(srcData.subarray(start, end), strOffset);
            strOffset += end - start;
          }
          row++;
        }
      }
      offsets[totalRows] = strOffset;
      columns.push({ name, dtype, data: (strBuf.buffer as ArrayBuffer).slice(0, strOffset), rowCount: totalRows, offsets });
    } else if (dtype === DTYPE_F32VEC) {
      const dim = decoded[0].columns[ci].vectorDim || 0;
      const buf = new ArrayBuffer(totalRows * dim * 4);
      const out = new Uint8Array(buf);
      let offset = 0;
      for (const batch of decoded) {
        const src = new Uint8Array(batch.columns[ci].data);
        out.set(src, offset);
        offset += src.length;
      }
      columns.push({ name, dtype, data: buf, rowCount: totalRows, vectorDim: dim });
    } else {
      columns.push({ name, dtype, data: new ArrayBuffer(0), rowCount: totalRows });
    }
  }

  return encodeColumnarBatch({ columns, rowCount: totalRows });
}

// ============================================================================
// Concat ColumnarBatches (stays columnar — no Row[])
// ============================================================================

/** Concatenate multiple ColumnarBatches with the same schema. */
export function concatColumnarBatches(batches: ColumnarBatch[]): ColumnarBatch | null {
  if (batches.length === 0) return null;
  if (batches.length === 1) return batches[0];

  const totalRows = batches.reduce((s, b) => s + b.rowCount, 0);
  const numCols = batches[0].columns.length;
  const columns: ColumnarColumn[] = [];

  for (let ci = 0; ci < numCols; ci++) {
    const dtype = batches[0].columns[ci].dtype;
    const name = batches[0].columns[ci].name;
    const bpe = bytesPerElement(dtype);

    if (bpe > 0) {
      const buf = new ArrayBuffer(totalRows * bpe);
      const out = new Uint8Array(buf);
      let offset = 0;
      for (const batch of batches) {
        const src = new Uint8Array(batch.columns[ci].data, 0, batch.rowCount * bpe);
        out.set(src, offset);
        offset += src.byteLength;
      }
      columns.push({ name, dtype, data: buf, rowCount: totalRows });
    } else if (dtype === DTYPE_BOOL) {
      const buf = new ArrayBuffer(Math.ceil(totalRows / 8));
      const out = new Uint8Array(buf);
      let row = 0;
      for (const batch of batches) {
        const src = new Uint8Array(batch.columns[ci].data);
        for (let r = 0; r < batch.rowCount; r++) {
          if (src[r >> 3] & (1 << (r & 7))) out[row >> 3] |= 1 << (row & 7);
          row++;
        }
      }
      columns.push({ name, dtype, data: buf, rowCount: totalRows });
    } else if (dtype === DTYPE_UTF8) {
      let totalStrLen = 0;
      for (const batch of batches) {
        const col = batch.columns[ci];
        totalStrLen += col.offsets ? col.offsets[batch.rowCount] : col.data.byteLength;
      }
      const offsets = new Uint32Array(totalRows + 1);
      const strBuf = new Uint8Array(totalStrLen);
      let strOffset = 0;
      let row = 0;
      for (const batch of batches) {
        const col = batch.columns[ci];
        const srcOffsets = col.offsets!;
        const srcData = new Uint8Array(col.data);
        for (let r = 0; r < batch.rowCount; r++) {
          offsets[row] = strOffset;
          const start = srcOffsets[r];
          const end = srcOffsets[r + 1];
          if (end > start) {
            strBuf.set(srcData.subarray(start, end), strOffset);
            strOffset += end - start;
          }
          row++;
        }
      }
      offsets[totalRows] = strOffset;
      columns.push({ name, dtype, data: (strBuf.buffer as ArrayBuffer).slice(0, strOffset), rowCount: totalRows, offsets });
    } else if (dtype === DTYPE_F32VEC) {
      const dim = batches[0].columns[ci].vectorDim || 0;
      const buf = new ArrayBuffer(totalRows * dim * 4);
      const out = new Uint8Array(buf);
      let offset = 0;
      for (const batch of batches) {
        const src = new Uint8Array(batch.columns[ci].data, 0, batch.rowCount * dim * 4);
        out.set(src, offset);
        offset += src.byteLength;
      }
      columns.push({ name, dtype, data: buf, rowCount: totalRows, vectorDim: dim });
    } else {
      columns.push({ name, dtype, data: new ArrayBuffer(0), rowCount: totalRows });
    }
  }

  return { columns, rowCount: totalRows };
}

// ============================================================================
// Columnar k-way merge (sorted merge without Row[])
// ============================================================================

/** Read a comparable value from a column at a given row index. */
export function readColumnValue(col: ColumnarColumn, row: number): number | bigint | string | boolean | null {
  if (col.nullBitmap && (col.nullBitmap[row >> 3] & (1 << (row & 7)))) return null;
  switch (col.dtype) {
    case DTYPE_F64: return new Float64Array(col.data)[row];
    case DTYPE_I64: return new BigInt64Array(col.data)[row];
    case DTYPE_I32: return new Int32Array(col.data)[row];
    case DTYPE_F32: return new Float32Array(col.data)[row];
    case DTYPE_UTF8: {
      const offsets = col.offsets!;
      const start = offsets[row], end = offsets[row + 1];
      return textDecoder.decode(new Uint8Array(col.data, start, end - start));
    }
    case DTYPE_BOOL: {
      const bits = new Uint8Array(col.data);
      return (bits[row >> 3] & (1 << (row & 7))) !== 0;
    }
    default: return null;
  }
}

/** Write a single row's value from src column at srcRow to dst column at dstRow. */
function copyColumnValue(
  dst: { data: Uint8Array; offsets?: number[]; strOffset: number },
  src: ColumnarColumn,
  srcRow: number,
  dstRow: number,
  dtype: number,
  bpe: number,
): void {
  if (bpe > 0) {
    // Fixed-width: copy bytes
    const srcBytes = new Uint8Array(src.data, srcRow * bpe, bpe);
    dst.data.set(srcBytes, dstRow * bpe);
  } else if (dtype === DTYPE_BOOL) {
    const srcBits = new Uint8Array(src.data);
    if (srcBits[srcRow >> 3] & (1 << (srcRow & 7))) {
      dst.data[dstRow >> 3] |= 1 << (dstRow & 7);
    }
  } else if (dtype === DTYPE_UTF8) {
    const srcOffsets = src.offsets!;
    const start = srcOffsets[srcRow];
    const end = srcOffsets[srcRow + 1];
    const len = end - start;
    dst.offsets![dstRow] = dst.strOffset;
    if (len > 0) {
      dst.data.set(new Uint8Array(src.data, start, len), dst.strOffset);
    }
    dst.strOffset += len;
  }
}

interface MergeHeapEntry {
  batchIdx: number;
  rowIdx: number;
  value: number | bigint | string | boolean | null;
}

function mergeSiftDown(
  heap: MergeHeapEntry[],
  i: number,
  asc: boolean,
): void {
  const size = heap.length;
  while (true) {
    let target = i;
    const left = 2 * i + 1;
    const right = 2 * i + 2;
    if (left < size && mergeCompare(heap[left], heap[target], asc) < 0) target = left;
    if (right < size && mergeCompare(heap[right], heap[target], asc) < 0) target = right;
    if (target === i) break;
    [heap[i], heap[target]] = [heap[target], heap[i]];
    i = target;
  }
}

function mergeCompare(a: MergeHeapEntry, b: MergeHeapEntry, asc: boolean): number {
  const va = a.value, vb = b.value;
  if (va === vb) return 0;
  if (va === null || va === undefined) return 1;
  if (vb === null || vb === undefined) return -1;
  const cmp = va < vb ? -1 : 1;
  return asc ? cmp : -cmp;
}

/**
 * Columnar k-way merge. Merges sorted ColumnarBatches by sortColumn
 * without creating Row[] objects. Returns a new ColumnarBatch.
 */
export function columnarKWayMerge(
  batches: ColumnarBatch[],
  sortColumn: string,
  dir: "asc" | "desc",
  limit: number,
): ColumnarBatch {
  const asc = dir === "asc";
  const numCols = batches[0]?.columns.length ?? 0;

  // Find sort column index in each batch
  const sortColIndices = batches.map(b => b.columns.findIndex(c => c.name === sortColumn));

  // Initialize heap
  const indices: number[] = new Array(batches.length).fill(0);
  const heap: MergeHeapEntry[] = [];
  for (let i = 0; i < batches.length; i++) {
    if (batches[i].rowCount > 0) {
      const sci = sortColIndices[i];
      heap.push({ batchIdx: i, rowIdx: 0, value: readColumnValue(batches[i].columns[sci], 0) });
      indices[i] = 1;
    }
  }
  for (let i = (heap.length >> 1) - 1; i >= 0; i--) mergeSiftDown(heap, i, asc);

  // Determine output size (upper bound)
  const totalRows = Math.min(limit, batches.reduce((s, b) => s + b.rowCount, 0));

  // Allocate output columns
  const outCols: { data: Uint8Array; offsets?: number[]; strOffset: number; dtype: number; bpe: number; name: string; vectorDim?: number }[] = [];
  for (let ci = 0; ci < numCols; ci++) {
    const col = batches[0].columns[ci];
    const bpe = bytesPerElement(col.dtype);
    if (col.dtype === DTYPE_UTF8) {
      // Estimate string buffer size: sum of all string data
      let totalStrLen = 0;
      for (const b of batches) totalStrLen += b.columns[ci].offsets ? b.columns[ci].offsets![b.rowCount] : b.columns[ci].data.byteLength;
      outCols.push({
        data: new Uint8Array(totalStrLen),
        offsets: new Array(totalRows + 1),
        strOffset: 0,
        dtype: col.dtype,
        bpe: 0,
        name: col.name,
      });
    } else if (col.dtype === DTYPE_BOOL) {
      outCols.push({
        data: new Uint8Array(Math.ceil(totalRows / 8)),
        strOffset: 0,
        dtype: col.dtype,
        bpe: 0,
        name: col.name,
      });
    } else if (col.dtype === DTYPE_F32VEC) {
      const dim = col.vectorDim || 0;
      outCols.push({
        data: new Uint8Array(totalRows * dim * 4),
        strOffset: 0,
        dtype: col.dtype,
        bpe: dim * 4,
        name: col.name,
        vectorDim: dim,
      });
    } else {
      outCols.push({
        data: new Uint8Array(totalRows * bpe),
        strOffset: 0,
        dtype: col.dtype,
        bpe,
        name: col.name,
      });
    }
  }

  let outRow = 0;
  while (heap.length > 0 && outRow < limit) {
    const top = heap[0];
    const batch = batches[top.batchIdx];

    // Copy all columns for this row
    for (let ci = 0; ci < numCols; ci++) {
      copyColumnValue(outCols[ci], batch.columns[ci], top.rowIdx, outRow, outCols[ci].dtype, outCols[ci].bpe);
    }
    outRow++;

    // Advance
    const bi = top.batchIdx;
    if (indices[bi] < batch.rowCount) {
      top.rowIdx = indices[bi];
      top.value = readColumnValue(batch.columns[sortColIndices[bi]], indices[bi]);
      indices[bi]++;
      mergeSiftDown(heap, 0, asc);
    } else {
      heap[0] = heap[heap.length - 1];
      heap.pop();
      if (heap.length > 0) mergeSiftDown(heap, 0, asc);
    }
  }

  // Build final ColumnarBatch
  const columns: ColumnarColumn[] = [];
  for (let ci = 0; ci < numCols; ci++) {
    const oc = outCols[ci];
    if (oc.dtype === DTYPE_UTF8) {
      oc.offsets![outRow] = oc.strOffset;
      columns.push({
        name: oc.name, dtype: oc.dtype, rowCount: outRow,
        data: (oc.data.buffer as ArrayBuffer).slice(0, oc.strOffset),
        offsets: new Uint32Array(oc.offsets!.slice(0, outRow + 1)),
      });
    } else if (oc.dtype === DTYPE_BOOL) {
      columns.push({ name: oc.name, dtype: oc.dtype, rowCount: outRow, data: (oc.data.buffer as ArrayBuffer).slice(0, Math.ceil(outRow / 8)) });
    } else if (oc.dtype === DTYPE_F32VEC) {
      const dim = oc.vectorDim || 0;
      columns.push({ name: oc.name, dtype: oc.dtype, rowCount: outRow, data: (oc.data.buffer as ArrayBuffer).slice(0, outRow * dim * 4), vectorDim: dim });
    } else {
      columns.push({ name: oc.name, dtype: oc.dtype, rowCount: outRow, data: (oc.data.buffer as ArrayBuffer).slice(0, outRow * oc.bpe) });
    }
  }

  return { columns, rowCount: outRow };
}

// ============================================================================
// Slice a ColumnarBatch (offset + limit without Row[])
// ============================================================================

/** Slice a ColumnarBatch by offset and limit. Returns a new batch (no Row[]). */
export function sliceColumnarBatch(batch: ColumnarBatch, offset: number, limit?: number): ColumnarBatch {
  const start = Math.min(offset, batch.rowCount);
  const end = limit !== undefined ? Math.min(start + limit, batch.rowCount) : batch.rowCount;
  const rowCount = end - start;
  if (rowCount <= 0) return { columns: batch.columns.map(c => ({ ...c, rowCount: 0, data: new ArrayBuffer(0) })), rowCount: 0 };
  if (start === 0 && end === batch.rowCount) return batch;

  const columns: ColumnarColumn[] = [];
  for (const col of batch.columns) {
    const bpe = bytesPerElement(col.dtype);
    if (bpe > 0) {
      columns.push({ name: col.name, dtype: col.dtype, rowCount, data: col.data.slice(start * bpe, end * bpe) });
    } else if (col.dtype === DTYPE_BOOL) {
      // Re-pack bits
      const src = new Uint8Array(col.data);
      const dst = new Uint8Array(Math.ceil(rowCount / 8));
      for (let r = 0; r < rowCount; r++) {
        const sr = start + r;
        if (src[sr >> 3] & (1 << (sr & 7))) dst[r >> 3] |= 1 << (r & 7);
      }
      columns.push({ name: col.name, dtype: col.dtype, rowCount, data: dst.buffer });
    } else if (col.dtype === DTYPE_UTF8) {
      const srcOffsets = col.offsets!;
      const strStart = srcOffsets[start];
      const strEnd = srcOffsets[end];
      const offsets = new Uint32Array(rowCount + 1);
      for (let r = 0; r <= rowCount; r++) offsets[r] = srcOffsets[start + r] - strStart;
      columns.push({ name: col.name, dtype: col.dtype, rowCount, data: col.data.slice(strStart, strEnd), offsets });
    } else if (col.dtype === DTYPE_F32VEC) {
      const dim = col.vectorDim || 0;
      columns.push({ name: col.name, dtype: col.dtype, rowCount, data: col.data.slice(start * dim * 4, end * dim * 4), vectorDim: dim });
    } else {
      columns.push({ name: col.name, dtype: col.dtype, rowCount, data: new ArrayBuffer(0) });
    }
  }
  return { columns, rowCount };
}

// ============================================================================
// Columnar → Row[] materialization (final boundary only)
// ============================================================================

/** Convert a ColumnarBatch to Row[] objects. Use only at response boundary. */
export function columnarBatchToRows(batch: ColumnarBatch): Row[] {
  const { columns, rowCount } = batch;
  const rows: Row[] = new Array(rowCount);
  for (let r = 0; r < rowCount; r++) rows[r] = {};

  for (const col of columns) {
    switch (col.dtype) {
      case DTYPE_F64: {
        const arr = new Float64Array(col.data);
        for (let r = 0; r < rowCount; r++) rows[r][col.name] = arr[r];
        break;
      }
      case DTYPE_I64: {
        const arr = new BigInt64Array(col.data);
        for (let r = 0; r < rowCount; r++) rows[r][col.name] = arr[r];
        break;
      }
      case DTYPE_I32: {
        const arr = new Int32Array(col.data);
        for (let r = 0; r < rowCount; r++) rows[r][col.name] = arr[r];
        break;
      }
      case DTYPE_F32: {
        const arr = new Float32Array(col.data);
        for (let r = 0; r < rowCount; r++) rows[r][col.name] = arr[r];
        break;
      }
      case DTYPE_UTF8: {
        const offsets = col.offsets!;
        const data = new Uint8Array(col.data);
        for (let r = 0; r < rowCount; r++) {
          const start = offsets[r];
          const end = offsets[r + 1];
          rows[r][col.name] = textDecoder.decode(data.subarray(start, end));
        }
        break;
      }
      case DTYPE_BOOL: {
        const bits = new Uint8Array(col.data);
        for (let r = 0; r < rowCount; r++) {
          rows[r][col.name] = (bits[r >> 3] & (1 << (r & 7))) !== 0;
        }
        break;
      }
      case DTYPE_F32VEC: {
        const dim = col.vectorDim!;
        const arr = new Float32Array(col.data);
        for (let r = 0; r < rowCount; r++) {
          rows[r][col.name] = arr.slice(r * dim, (r + 1) * dim);
        }
        break;
      }
      default:
        for (let r = 0; r < rowCount; r++) rows[r][col.name] = null;
        break;
    }

    if (col.nullBitmap) {
      for (let r = 0; r < rowCount; r++) {
        if (col.nullBitmap[r >> 3] & (1 << (r & 7))) rows[r][col.name] = null;
      }
    }
  }

  return rows;
}
