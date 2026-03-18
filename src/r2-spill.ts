/**
 * Columnar binary spill storage for sort/join operators.
 *
 * Data is stored in a compact columnar binary format (not row-based JSON).
 * Each run is a self-contained columnar file:
 *
 *   Header:
 *     [4 bytes] magic "QMCB" (QueryMode Columnar Binary)
 *     [4 bytes] rowCount (uint32)
 *     [2 bytes] columnCount (uint16)
 *
 *   Per-column descriptor:
 *     [2 bytes] nameLength (uint16)
 *     [nameLength bytes] name (UTF-8)
 *     [1 byte] dtype tag (0=f64, 1=i64, 2=utf8, 3=bool, 4=f32vec, 5=null_only)
 *     [1 byte] hasNulls (0 or 1)
 *
 *   Per-column data (same order as descriptors):
 *     if hasNulls: [ceil(rowCount/8) bytes] null bitmap
 *     dtype-specific payload:
 *       f64:     rowCount × 8 bytes (Float64Array)
 *       i64:     rowCount × 8 bytes (BigInt64Array)
 *       utf8:    4 bytes totalLen + (rowCount+1)×4 offsets + data
 *       bool:    ceil(rowCount/8) bytes packed bits
 *       f32vec:  4 bytes dimension + rowCount×dimension×4 bytes
 *       null_only: 0 bytes
 *
 * R2SpillBackend stores these under `__spill/{prefix}/` and cleans up
 * in a finally block — no orphaned objects.
 */

import type { Row } from "./types.js";
import { withRetry, withTimeout } from "./coalesce.js";

const R2_SPILL_TIMEOUT_MS = 10_000;
const MAGIC = 0x42434D51; // "QMCB" little-endian
const encoder = new TextEncoder();
const decoder = new TextDecoder();

// Dtype tags
const DTYPE_F64 = 0;
const DTYPE_I64 = 1;
const DTYPE_UTF8 = 2;
const DTYPE_BOOL = 3;
const DTYPE_F32VEC = 4;
const DTYPE_NULL = 5;

/** Generic spill backend for sort/join operators. */
export interface SpillBackend {
  /** Write a sorted run of rows. Returns a spill ID for later streaming. */
  writeRun(rows: Row[]): Promise<string>;
  /** Stream rows back from a previously written run. */
  streamRun(spillId: string): AsyncGenerator<Row>;
  /** Total bytes written so far. */
  bytesWritten: number;
  /** Total bytes read so far. */
  bytesRead: number;
  /** Delete all spill data. Safe to call multiple times. */
  cleanup(): Promise<void>;
}

/** Infer dtype tag from a JS value. */
function inferDtype(value: unknown): number {
  if (value === null || value === undefined) return DTYPE_NULL;
  if (typeof value === "number") return DTYPE_F64;
  if (typeof value === "bigint") return DTYPE_I64;
  if (typeof value === "string") return DTYPE_UTF8;
  if (typeof value === "boolean") return DTYPE_BOOL;
  if (value instanceof Float32Array) return DTYPE_F32VEC;
  return DTYPE_F64; // fallback
}

/**
 * Encode rows into columnar binary format.
 * Returns a single ArrayBuffer containing the entire run.
 */
export function encodeColumnarRun(rows: Row[]): ArrayBuffer {
  const rowCount = rows.length;
  if (rowCount === 0) {
    // Empty run: just header
    const buf = new ArrayBuffer(10);
    const view = new DataView(buf);
    view.setUint32(0, MAGIC, true);
    view.setUint32(4, 0, true);
    view.setUint16(8, 0, true);
    return buf;
  }

  // Discover columns: union of all keys, infer dtype from first non-null value
  const colNames: string[] = [];
  const colNameSet = new Set<string>();
  const colDtypes: number[] = [];

  for (const row of rows) {
    for (const key in row) {
      if (!colNameSet.has(key)) {
        colNameSet.add(key);
        colNames.push(key);
        colDtypes.push(DTYPE_NULL); // will be resolved below
      }
    }
  }

  // Resolve dtypes from first non-null value per column
  for (let ci = 0; ci < colNames.length; ci++) {
    if (colDtypes[ci] !== DTYPE_NULL) continue;
    const name = colNames[ci];
    for (const row of rows) {
      const val = row[name];
      if (val !== null && val !== undefined) {
        colDtypes[ci] = inferDtype(val);
        break;
      }
    }
  }

  const columnCount = colNames.length;

  // Pre-encode column names
  const encodedNames: Uint8Array[] = colNames.map(n => encoder.encode(n));

  // Check for nulls per column and build null bitmaps
  const nullBitmaps: Uint8Array[] = [];
  const hasNulls: boolean[] = [];
  const bitmapBytes = Math.ceil(rowCount / 8);

  for (let ci = 0; ci < columnCount; ci++) {
    const name = colNames[ci];
    let anyNull = false;
    const bitmap = new Uint8Array(bitmapBytes);

    for (let ri = 0; ri < rowCount; ri++) {
      const val = rows[ri][name];
      if (val === null || val === undefined) {
        anyNull = true;
        bitmap[ri >> 3] |= 1 << (ri & 7);
      }
    }

    hasNulls.push(anyNull);
    nullBitmaps.push(anyNull ? bitmap : new Uint8Array(0));
  }

  // Calculate total size
  // Header: 4 + 4 + 2 = 10
  let totalSize = 10;

  // Column descriptors
  for (let ci = 0; ci < columnCount; ci++) {
    totalSize += 2 + encodedNames[ci].length + 1 + 1; // nameLen + name + dtype + hasNulls
  }

  // Pre-encode UTF8 strings once (reused in write pass to avoid double-encoding)
  const preEncodedUtf8 = new Map<number, Uint8Array[]>();

  // Column data
  for (let ci = 0; ci < columnCount; ci++) {
    if (hasNulls[ci]) totalSize += bitmapBytes;

    switch (colDtypes[ci]) {
      case DTYPE_F64: totalSize += rowCount * 8; break;
      case DTYPE_I64: totalSize += rowCount * 8; break;
      case DTYPE_BOOL: totalSize += bitmapBytes; break;
      case DTYPE_UTF8: {
        // offsets + string data
        totalSize += 4; // totalLen
        totalSize += (rowCount + 1) * 4; // offsets
        const name = colNames[ci];
        const encoded: Uint8Array[] = [];
        let strBytes = 0;
        for (let ri = 0; ri < rowCount; ri++) {
          const val = rows[ri][name];
          const enc = typeof val === "string" ? encoder.encode(val) : new Uint8Array(0);
          encoded.push(enc);
          strBytes += enc.length;
        }
        preEncodedUtf8.set(ci, encoded);
        totalSize += strBytes;
        break;
      }
      case DTYPE_F32VEC: {
        totalSize += 4; // dimension
        const name = colNames[ci];
        let dim = 0;
        for (const row of rows) {
          const val = row[name];
          if (val instanceof Float32Array) { dim = val.length; break; }
        }
        totalSize += rowCount * dim * 4;
        break;
      }
      case DTYPE_NULL: break; // no data
    }
  }

  // Allocate buffer and write
  const buf = new ArrayBuffer(totalSize);
  const view = new DataView(buf);
  const bytes = new Uint8Array(buf);
  let offset = 0;

  // Header
  view.setUint32(offset, MAGIC, true); offset += 4;
  view.setUint32(offset, rowCount, true); offset += 4;
  view.setUint16(offset, columnCount, true); offset += 2;

  // Column descriptors
  for (let ci = 0; ci < columnCount; ci++) {
    const nameBytes = encodedNames[ci];
    view.setUint16(offset, nameBytes.length, true); offset += 2;
    bytes.set(nameBytes, offset); offset += nameBytes.length;
    view.setUint8(offset, colDtypes[ci]); offset += 1;
    view.setUint8(offset, hasNulls[ci] ? 1 : 0); offset += 1;
  }

  // Column data
  for (let ci = 0; ci < columnCount; ci++) {
    const name = colNames[ci];
    const dtype = colDtypes[ci];

    // Null bitmap
    if (hasNulls[ci]) {
      bytes.set(nullBitmaps[ci], offset);
      offset += bitmapBytes;
    }

    switch (dtype) {
      case DTYPE_F64: {
        for (let ri = 0; ri < rowCount; ri++) {
          const val = rows[ri][name];
          view.setFloat64(offset + ri * 8, typeof val === "number" ? val : 0, true);
        }
        offset += rowCount * 8;
        break;
      }
      case DTYPE_I64: {
        for (let ri = 0; ri < rowCount; ri++) {
          const val = rows[ri][name];
          view.setBigInt64(offset + ri * 8, typeof val === "bigint" ? val : 0n, true);
        }
        offset += rowCount * 8;
        break;
      }
      case DTYPE_BOOL: {
        const packed = new Uint8Array(buf, offset, bitmapBytes);
        for (let ri = 0; ri < rowCount; ri++) {
          if (rows[ri][name] === true) packed[ri >> 3] |= 1 << (ri & 7);
        }
        offset += bitmapBytes;
        break;
      }
      case DTYPE_UTF8: {
        // Reuse pre-encoded strings from size calculation pass
        const encodedStrs = preEncodedUtf8.get(ci)!;
        let totalLen = 0;
        for (let ri = 0; ri < rowCount; ri++) totalLen += encodedStrs[ri].length;

        view.setUint32(offset, totalLen, true); offset += 4;

        // Offsets array (use DataView to avoid alignment issues)
        let strOff = 0;
        for (let ri = 0; ri < rowCount; ri++) {
          view.setUint32(offset + ri * 4, strOff, true);
          strOff += encodedStrs[ri].length;
        }
        view.setUint32(offset + rowCount * 4, strOff, true);
        offset += (rowCount + 1) * 4;

        // String data
        for (let ri = 0; ri < rowCount; ri++) {
          bytes.set(encodedStrs[ri], offset);
          offset += encodedStrs[ri].length;
        }
        break;
      }
      case DTYPE_F32VEC: {
        let dim = 0;
        for (const row of rows) {
          const val = row[name];
          if (val instanceof Float32Array) { dim = val.length; break; }
        }
        view.setUint32(offset, dim, true); offset += 4;

        for (let ri = 0; ri < rowCount; ri++) {
          const val = rows[ri][name];
          if (val instanceof Float32Array) {
            bytes.set(new Uint8Array(val.buffer, val.byteOffset, val.byteLength), offset);
          }
          // else leave as zeros for null
          offset += dim * 4;
        }
        break;
      }
      case DTYPE_NULL:
        break;
    }
  }

  return buf;
}

/** Column descriptor decoded from the header. */
interface ColumnDescriptor {
  name: string;
  dtype: number;
  hasNulls: boolean;
}

/**
 * Decode a columnar binary run back to rows.
 * Yields rows one at a time for streaming consumption by k-way merge.
 */
export function* decodeColumnarRun(buf: ArrayBuffer): Generator<Row> {
  const view = new DataView(buf);
  const bytes = new Uint8Array(buf);

  const magic = view.getUint32(0, true);
  if (magic !== MAGIC) throw new Error("Invalid spill file: bad magic");

  const rowCount = view.getUint32(4, true);
  const columnCount = view.getUint16(8, true);
  if (rowCount === 0) return;

  let offset = 10;

  // Read column descriptors
  const descriptors: ColumnDescriptor[] = [];
  for (let ci = 0; ci < columnCount; ci++) {
    const nameLen = view.getUint16(offset, true); offset += 2;
    const name = decoder.decode(bytes.subarray(offset, offset + nameLen)); offset += nameLen;
    const dtype = view.getUint8(offset); offset += 1;
    const hasNulls = view.getUint8(offset) === 1; offset += 1;
    descriptors.push({ name, dtype, hasNulls });
  }

  const bitmapBytes = Math.ceil(rowCount / 8);

  // Read column data into arrays
  const colArrays: {
    nullBitmap: Uint8Array | null;
    f64?: Float64Array;
    i64?: BigInt64Array;
    boolBits?: Uint8Array;
    strOffsets?: Uint32Array;
    strData?: Uint8Array;
    f32dim?: number;
    f32data?: Uint8Array;
  }[] = [];

  for (let ci = 0; ci < columnCount; ci++) {
    const desc = descriptors[ci];
    let nullBitmap: Uint8Array | null = null;

    if (desc.hasNulls) {
      nullBitmap = bytes.subarray(offset, offset + bitmapBytes);
      offset += bitmapBytes;
    }

    const entry: (typeof colArrays)[0] = { nullBitmap };

    switch (desc.dtype) {
      case DTYPE_F64: {
        // Need aligned copy for Float64Array
        const aligned = new Float64Array(rowCount);
        const src = new Uint8Array(buf, offset, rowCount * 8);
        new Uint8Array(aligned.buffer).set(src);
        entry.f64 = aligned;
        offset += rowCount * 8;
        break;
      }
      case DTYPE_I64: {
        const aligned = new BigInt64Array(rowCount);
        const src = new Uint8Array(buf, offset, rowCount * 8);
        new Uint8Array(aligned.buffer).set(src);
        entry.i64 = aligned;
        offset += rowCount * 8;
        break;
      }
      case DTYPE_BOOL: {
        entry.boolBits = bytes.subarray(offset, offset + bitmapBytes);
        offset += bitmapBytes;
        break;
      }
      case DTYPE_UTF8: {
        const totalLen = view.getUint32(offset, true); offset += 4;
        // Aligned copy for Uint32Array offsets
        const offsetsAligned = new Uint32Array(rowCount + 1);
        const offsetsSrc = new Uint8Array(buf, offset, (rowCount + 1) * 4);
        new Uint8Array(offsetsAligned.buffer).set(offsetsSrc);
        entry.strOffsets = offsetsAligned;
        offset += (rowCount + 1) * 4;
        entry.strData = bytes.subarray(offset, offset + totalLen);
        offset += totalLen;
        break;
      }
      case DTYPE_F32VEC: {
        const dim = view.getUint32(offset, true); offset += 4;
        entry.f32dim = dim;
        entry.f32data = bytes.subarray(offset, offset + rowCount * dim * 4);
        offset += rowCount * dim * 4;
        break;
      }
      case DTYPE_NULL:
        break;
    }

    colArrays.push(entry);
  }

  // Yield rows one at a time
  for (let ri = 0; ri < rowCount; ri++) {
    const row: Row = {};
    for (let ci = 0; ci < columnCount; ci++) {
      const desc = descriptors[ci];
      const col = colArrays[ci];

      // Check null bitmap
      if (col.nullBitmap && (col.nullBitmap[ri >> 3] & (1 << (ri & 7)))) {
        row[desc.name] = null;
        continue;
      }

      switch (desc.dtype) {
        case DTYPE_F64:
          row[desc.name] = col.f64![ri];
          break;
        case DTYPE_I64:
          row[desc.name] = col.i64![ri];
          break;
        case DTYPE_BOOL:
          row[desc.name] = !!(col.boolBits![ri >> 3] & (1 << (ri & 7)));
          break;
        case DTYPE_UTF8: {
          const start = col.strOffsets![ri];
          const end = col.strOffsets![ri + 1];
          row[desc.name] = decoder.decode(col.strData!.subarray(start, end));
          break;
        }
        case DTYPE_F32VEC: {
          const dim = col.f32dim!;
          const byteOff = ri * dim * 4;
          const vecBytes = col.f32data!.subarray(byteOff, byteOff + dim * 4);
          const vec = new Float32Array(dim);
          new Uint8Array(vec.buffer).set(vecBytes);
          row[desc.name] = vec;
          break;
        }
        case DTYPE_NULL:
          row[desc.name] = null;
          break;
      }
    }
    yield row;
  }
}

/**
 * R2-backed spill backend for Cloudflare Workers edge.
 * Stores columnar binary files under `__spill/{prefix}/{runIndex}.bin`.
 */
export class R2SpillBackend implements SpillBackend {
  private bucket: R2Bucket;
  private prefix: string;
  private runCount = 0;
  private spillKeys: string[] = [];
  bytesWritten = 0;
  bytesRead = 0;

  constructor(bucket: R2Bucket, prefix: string) {
    this.bucket = bucket;
    this.prefix = prefix;
  }

  async writeRun(rows: Row[]): Promise<string> {
    const key = `${this.prefix}/${this.runCount++}.bin`;
    const buf = encodeColumnarRun(rows);

    await withRetry(() =>
      withTimeout(this.bucket.put(key, buf), R2_SPILL_TIMEOUT_MS),
    );

    this.bytesWritten += buf.byteLength;
    this.spillKeys.push(key);
    return key;
  }

  async *streamRun(spillId: string): AsyncGenerator<Row> {
    const obj = await withRetry(() =>
      withTimeout(this.bucket.get(spillId), R2_SPILL_TIMEOUT_MS),
    );
    if (!obj) throw new Error(`Spill object not found: ${spillId}`);

    const buf = await obj.arrayBuffer();
    this.bytesRead += buf.byteLength;

    yield* decodeColumnarRun(buf);
  }

  async cleanup(): Promise<void> {
    const keys = this.spillKeys.splice(0);
    await Promise.all(
      keys.map(key => this.bucket.delete(key).catch(() => {})),
    );
  }
}
