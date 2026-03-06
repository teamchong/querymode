/**
 * Lance v2 format parser — shared between query-do.ts and local-executor.ts.
 *
 * Lance v2 uses a different protobuf structure for column metadata than v1.
 * The standard parseColumnMetaFromProtobuf() returns 0 pages and generic
 * column_N names for v2 files. This module provides correct v2 parsing.
 *
 * Layout of a Lance v2 file:
 *   [data buffers with alignment padding]
 *   [column metadata protobuf region]
 *   [column metadata offset table: numColumns * u64]
 *   [global buffer offset table: (numGlobalBuffers+1) * u64]
 *   [footer: 40 bytes]
 *
 * Footer (last 40 bytes):
 *   [columnMetaStart: u64] [columnMetaOffsetsStart: u64]
 *   [globalBufOffsetsStart: u64] [numGlobalBuffers: u32]
 *   [numColumns: u32] [majorVersion: u16] [minorVersion: u16] [magic: u32]
 *
 * Column metadata protobuf (per column, may have multiple page descriptors):
 *   field 1 (LEN) = encoding descriptor (e.g. "/lance.encodings.ColumnEncoding")
 *   field 2 (LEN) = page descriptor sub-message:
 *     field 1 (LEN) = buffer position(s) as packed varints
 *     field 2 (LEN) = data length info (for variable-length types)
 *     field 3 (VARINT) = row count
 *     field 4 (LEN) = array encoding descriptor
 *
 * For fixed-width columns (int64, float64, etc.), a single page descriptor
 * encodes the buffer position of the data. For variable-width columns (utf8),
 * there are two page descriptors: one for the offsets array and one whose
 * sub-pages contain the offsets buffer + data buffer positions.
 */

import type { ColumnMeta, DataType, SchemaField } from "./types.js";
import { logicalTypeToDataType } from "./manifest.js";

export interface LanceV2ColumnInfo {
  name: string;
  dtype: DataType;
  rowCount: number;
  /** Byte offset of data in the file */
  byteOffset: number;
  /** Byte length of data in the file */
  byteLength: number;
  /** For nullable columns: byte offset of validity bitmap */
  bitmapOffset?: number;
  /** For nullable columns: byte length of validity bitmap */
  bitmapLength?: number;
  /** For utf8: byte offset of the string offsets array */
  offsetsOffset?: number;
  /** For utf8: byte length of the string offsets array */
  offsetsLength?: number;
  /** For utf8: byte offset of the string data buffer */
  dataOffset?: number;
  /** For utf8: byte length of the string data buffer */
  dataLength?: number;
}

/** Read a varint from bytes, returning value and number of bytes consumed. */
function readLEB128(bytes: Uint8Array, pos: number): { val: number; len: number } {
  let val = 0, shift = 0, len = 0;
  while (pos + len < bytes.length) {
    const b = bytes[pos + len++];
    if (shift < 32) val |= (b & 0x7f) << shift;
    shift += 7;
    if (!(b & 0x80)) break;
  }
  return { val: val >>> 0, len };
}

/** Read all varints packed in a LEN-delimited field. */
function readPackedVarints(bytes: Uint8Array, start: number, length: number): number[] {
  const vals: number[] = [];
  let pos = start;
  const end = start + length;
  while (pos < end) {
    const { val, len } = readLEB128(bytes, pos);
    vals.push(val);
    pos += len;
  }
  return vals;
}

/** Parse a page descriptor sub-message. Returns buffer positions, data lengths, and row count. */
function parsePageDescriptor(bytes: Uint8Array, start: number, length: number): {
  bufferPositions: number[];
  dataLengths: number[];
  dataLength: number;
  rowCount: number;
} {
  let bufferPositions: number[] = [];
  let dataLengths: number[] = [];
  let rowCount = 0;
  let pos = start;
  const end = start + length;

  while (pos < end) {
    const tag = bytes[pos++];
    const fn = tag >> 3, wt = tag & 7;
    if (wt === 2) {
      const { val: len, len: lenBytes } = readLEB128(bytes, pos);
      pos += lenBytes;
      if (fn === 1) {
        bufferPositions = readPackedVarints(bytes, pos, len);
      } else if (fn === 2) {
        // Data length info — packed varints [bitmap_len, data_len] or [data_len]
        dataLengths = readPackedVarints(bytes, pos, len);
      }
      pos += len;
    } else if (wt === 0) {
      const { val, len } = readLEB128(bytes, pos);
      pos += len;
      if (fn === 3) rowCount = val;
    } else if (wt === 1) pos += 8;
    else if (wt === 5) pos += 4;
    else break;
  }

  return {
    bufferPositions,
    dataLengths,
    dataLength: dataLengths.length > 0 ? dataLengths[dataLengths.length - 1] : 0,
    rowCount,
  };
}

/**
 * Parse Lance v2 column metadata from a file's raw bytes.
 *
 * Lance v2 stores column metadata as a sequence of protobuf messages in the
 * region [colMetaStart, colMetaOffsetsStart). Each logical schema column
 * produces one or more physical column entries. For example, a utf8 column
 * produces both an offsets column and a data column.
 *
 * Instead of relying on the column offset table (which may point into the
 * data region for secondary columns), we parse all column proto messages
 * sequentially from colMetaStart and match them to schema fields.
 *
 * @param fileData - Full file contents as ArrayBuffer
 * @param schema - Optional manifest schema for column name/type resolution
 * @param fallbackTotalRows - Fallback row count from manifest (if protobuf doesn't encode it)
 * @returns Array of column info, or null if not a valid Lance v2 file
 */
export function parseLanceV2Columns(
  fileData: ArrayBuffer,
  schema?: SchemaField[],
  fallbackTotalRows?: number,
): LanceV2ColumnInfo[] | null {
  if (fileData.byteLength < 40) return null;

  const fileBytes = new Uint8Array(fileData);
  const footerStart = fileBytes.length - 40;

  // Read footer fields
  const footerDV = new DataView(fileData, footerStart, 40);
  const colMetaStart = Number(footerDV.getBigUint64(0, true));
  const colMetaOffsetsStart = Number(footerDV.getBigUint64(8, true));
  const numCols = footerDV.getUint32(28, true);

  if (numCols === 0 || numCols > 10000) return null;

  // Resolve leaf schema fields for name/type lookup
  const leafFields = schema?.filter(f => f.parentId === -1 || f.parentId === 0) ?? [];

  // Parse ALL column proto messages sequentially from the metadata region.
  // Each column proto consists of repeated (field 1: encoding, field 2: page) pairs.
  // A new column starts with field 1 after a field 2. We detect boundaries by
  // tracking field number transitions.
  type ColProto = { pages: ReturnType<typeof parsePageDescriptor>[] };
  const colProtos: ColProto[] = [];
  let currentCol: ColProto = { pages: [] };
  let lastFieldNum = 0;
  let pos = colMetaStart;

  while (pos < colMetaOffsetsStart) {
    if (pos >= fileBytes.length) break;
    const tag = fileBytes[pos++];
    const fn = tag >> 3, wt = tag & 7;

    // A field 1 after a field 2 signals the start of a new column
    if (fn === 1 && lastFieldNum === 2 && currentCol.pages.length > 0) {
      colProtos.push(currentCol);
      currentCol = { pages: [] };
    }

    if (wt === 2) {
      const { val: len, len: lenBytes } = readLEB128(fileBytes, pos);
      pos += lenBytes;
      if (fn === 2) {
        currentCol.pages.push(parsePageDescriptor(fileBytes, pos, len));
      }
      pos += len;
    } else if (wt === 0) {
      const { len } = readLEB128(fileBytes, pos);
      pos += len;
    } else if (wt === 1) pos += 8;
    else if (wt === 5) pos += 4;
    else break;

    lastFieldNum = fn;
  }
  // Push the last column
  if (currentCol.pages.length > 0) colProtos.push(currentCol);

  // Now map physical columns to logical schema columns.
  // For fixed-width types: 1 physical column per logical column
  // For utf8/binary: 2 physical columns (offsets column + data column) per logical column
  const colInfos: LanceV2ColumnInfo[] = [];
  let physIdx = 0;

  for (let logIdx = 0; logIdx < (leafFields.length || numCols); logIdx++) {
    if (physIdx >= colProtos.length) break;

    const name = leafFields[logIdx]?.name ?? `column_${logIdx}`;
    const dtype: DataType = leafFields[logIdx]
      ? logicalTypeToDataType(leafFields[logIdx].logicalType)
      : "int64";

    // Get row count from first available page descriptor
    let rowCount = 0;
    for (const p of colProtos[physIdx].pages) {
      if (p.rowCount > 0) { rowCount = p.rowCount; break; }
    }
    if (rowCount === 0 && fallbackTotalRows) rowCount = fallbackTotalRows;

    if (dtype === "utf8" || dtype === "binary") {
      // UTF8 columns can be encoded two ways in Lance v2:
      // A) Single physical column with 2 buffer positions: [offsets_pos, data_pos]
      // B) Two physical columns: offsets column + data column
      const proto = colProtos[physIdx];
      const firstPage = proto.pages[0];
      const bufPositions = firstPage?.bufferPositions ?? [];

      let offsetsBufPos: number;
      let dataBufPos: number;
      let dataBufLen: number;

      if (bufPositions.length >= 2) {
        // Case A: single proto with both positions
        offsetsBufPos = bufPositions[0];
        dataBufPos = bufPositions[1];
        dataBufLen = firstPage?.dataLength ?? 0;
        physIdx += 1;
      } else {
        // Case B: two physical columns
        offsetsBufPos = bufPositions[0] ?? 0;
        const dataProto = physIdx + 1 < colProtos.length ? colProtos[physIdx + 1] : null;
        if (dataProto) {
          const dp = dataProto.pages[0];
          dataBufPos = dp?.bufferPositions[0] ?? (offsetsBufPos + rowCount * 8);
          dataBufLen = dp?.dataLength ?? 0;
          // Get row count from data proto if not available
          if (rowCount === 0) {
            for (const p of dataProto.pages) {
              if (p.rowCount > 0) { rowCount = p.rowCount; break; }
            }
          }
          physIdx += 2;
        } else {
          dataBufPos = offsetsBufPos + rowCount * 8;
          dataBufLen = 0;
          physIdx += 1;
        }
      }

      if (rowCount === 0 && fallbackTotalRows) rowCount = fallbackTotalRows;
      const offsetsLen = rowCount * 8;

      colInfos.push({
        name, dtype, rowCount,
        byteOffset: offsetsBufPos,
        byteLength: dataBufLen > 0 ? (dataBufPos - offsetsBufPos + dataBufLen) : offsetsLen,
        offsetsOffset: offsetsBufPos,
        offsetsLength: offsetsLen,
        dataOffset: dataBufPos,
        dataLength: dataBufLen,
      });

    } else {
      // Fixed-width column — may be nullable (2 buffer positions: bitmap + data)
      const proto = colProtos[physIdx];
      const firstPage = proto.pages[0];
      const bufPositions = firstPage?.bufferPositions ?? [];
      const dataLengths = firstPage?.dataLengths ?? [];
      const bpv = bytesPerValue(dtype);

      if (bufPositions.length >= 2) {
        // Nullable column: [bitmap_offset, data_offset]
        const bitmapOff = bufPositions[0];
        const dataOff = bufPositions[1];
        const bitmapLen = dataLengths[0] ?? Math.ceil(rowCount / 8);
        const dataLen = dataLengths[1] ?? (rowCount * bpv);

        colInfos.push({
          name, dtype, rowCount,
          byteOffset: dataOff,
          byteLength: dataLen,
          bitmapOffset: bitmapOff,
          bitmapLength: bitmapLen,
        });
      } else {
        // Non-nullable: single buffer position
        const bufPos = bufPositions[0] ?? 0;
        colInfos.push({
          name, dtype, rowCount,
          byteOffset: bufPos,
          byteLength: rowCount * bpv,
        });
      }

      physIdx += 1;
    }
  }

  return colInfos;
}

function bytesPerValue(dtype: DataType): number {
  switch (dtype) {
    case "int8": case "uint8": case "bool": return 1;
    case "int16": case "uint16": case "float16": return 2;
    case "int32": case "uint32": case "float32": return 4;
    default: return 8;
  }
}

/**
 * Convert Lance v2 column info to ColumnMeta with page info.
 * Uses actual byte offsets extracted from the protobuf metadata.
 *
 * For utf8/binary columns with offsets+data layout, the page covers
 * both the offsets array and data buffer contiguously by spanning from
 * offsetsOffset to dataOffset+dataLength. The decodeLanceV2Utf8 function
 * separates them during decoding using the known offsets array size.
 */
export function lanceV2ToColumnMeta(colInfos: LanceV2ColumnInfo[]): ColumnMeta[] {
  const columns: ColumnMeta[] = [];

  for (const col of colInfos) {
    if ((col.dtype === "utf8" || col.dtype === "binary") &&
        col.offsetsOffset !== undefined && col.dataOffset !== undefined &&
        col.dataLength !== undefined && col.dataLength > 0) {
      // For utf8: create page spanning from offsets start through data end.
      // The decoder uses rowCount to find the offsets/data boundary.
      const totalStart = col.offsetsOffset;
      const totalEnd = col.dataOffset + col.dataLength;
      columns.push({
        name: col.name,
        dtype: col.dtype,
        nullCount: 0,
        pages: [{
          byteOffset: BigInt(totalStart),
          byteLength: totalEnd - totalStart,
          rowCount: col.rowCount,
          nullCount: 0,
        }],
      });
    } else if (col.bitmapOffset !== undefined && col.bitmapLength !== undefined && col.bitmapLength > 0) {
      // Nullable fixed-width: page spans from bitmap start through data end.
      // Lance v2 uses 64-byte alignment between bitmap and data, so we store
      // the actual data offset within the page buffer so decodePage can skip
      // the alignment padding correctly.
      const dataOffsetInPage = col.byteOffset - col.bitmapOffset;
      columns.push({
        name: col.name,
        dtype: col.dtype,
        nullCount: 1, // non-zero signals nullable; actual count computed from bitmap during decode
        pages: [{
          byteOffset: BigInt(col.bitmapOffset),
          byteLength: col.byteLength + dataOffsetInPage,
          rowCount: col.rowCount,
          nullCount: 1,
          dataOffsetInPage,
        }],
      });
    } else {
      columns.push({
        name: col.name,
        dtype: col.dtype,
        nullCount: 0,
        pages: [{
          byteOffset: BigInt(col.byteOffset),
          byteLength: col.byteLength,
          rowCount: col.rowCount,
          nullCount: 0,
        }],
      });
    }
  }

  return columns;
}

/**
 * Compute min/max statistics for Lance v2 columns by reading page data.
 * Lance v2 protobuf doesn't store page-level stats, so we compute them
 * on first access and populate the ColumnMeta pages in place.
 *
 * Only computes stats for numeric types (int8-int64, uint8-uint64, float32, float64)
 * since those benefit most from page skipping.
 *
 * @param columns - ColumnMeta array to populate with stats (mutated in place)
 * @param readPage - Function to read a page's raw bytes given offset and length
 */
export async function computeLanceV2Stats(
  columns: ColumnMeta[],
  readPage: (byteOffset: number, byteLength: number) => Promise<ArrayBuffer>,
): Promise<void> {
  const numericTypes = new Set([
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float32", "float64",
  ]);

  for (const col of columns) {
    if (!numericTypes.has(col.dtype)) continue;

    for (const page of col.pages) {
      // Skip if stats already populated
      if (page.minValue !== undefined) continue;

      try {
        const buf = await readPage(Number(page.byteOffset), page.byteLength);
        const stats = computePageStats(buf, col.dtype, page.rowCount, page.nullCount, page.dataOffsetInPage);
        if (stats) {
          page.minValue = stats.min;
          page.maxValue = stats.max;
        }
      } catch {
        // Silently skip — stats are an optimization, not required
      }
    }
  }
}

/** Compute min/max from a raw page buffer for a numeric dtype. */
function computePageStats(
  buf: ArrayBuffer,
  dtype: string,
  rowCount: number,
  nullCount: number,
  dataOffsetInPage?: number,
): { min: number | bigint; max: number | bigint } | null {
  let dataBuf = buf;

  // Strip null bitmap for nullable columns
  if (nullCount > 0 && rowCount > 0) {
    const stripBytes = dataOffsetInPage ?? Math.ceil(rowCount / 8);
    dataBuf = buf.slice(stripBytes);
  }

  const view = new DataView(dataBuf);
  const numValues = dataOffsetInPage !== undefined ? rowCount : rowCount - nullCount;
  if (numValues <= 0) return null;

  // Read null bitmap to skip null positions
  let nullSet: Set<number> | null = null;
  if (nullCount > 0 && rowCount > 0) {
    const bitmapBytes = Math.ceil(rowCount / 8);
    if (buf.byteLength >= bitmapBytes) {
      const bytes = new Uint8Array(buf, 0, bitmapBytes);
      nullSet = new Set<number>();
      let idx = 0;
      for (let b = 0; b < bitmapBytes && idx < rowCount; b++) {
        for (let bit = 0; bit < 8 && idx < rowCount; bit++, idx++) {
          if (((bytes[b] >> bit) & 1) === 0) nullSet.add(idx);
        }
      }
    }
  }

  switch (dtype) {
    case "float64": {
      let min = Infinity, max = -Infinity;
      for (let i = 0; i < numValues; i++) {
        if (nullSet?.has(i)) continue;
        const v = view.getFloat64(i * 8, true);
        if (v < min) min = v;
        if (v > max) max = v;
      }
      return min <= max ? { min, max } : null;
    }
    case "float32": {
      let min = Infinity, max = -Infinity;
      for (let i = 0; i < numValues; i++) {
        if (nullSet?.has(i)) continue;
        const v = view.getFloat32(i * 4, true);
        if (v < min) min = v;
        if (v > max) max = v;
      }
      return min <= max ? { min, max } : null;
    }
    case "int64": {
      let min = BigInt("9223372036854775807"), max = BigInt("-9223372036854775808");
      for (let i = 0; i < numValues; i++) {
        if (nullSet?.has(i)) continue;
        const v = view.getBigInt64(i * 8, true);
        if (v < min) min = v;
        if (v > max) max = v;
      }
      return min <= max ? { min, max } : null;
    }
    case "int32": {
      let min = 2147483647, max = -2147483648;
      for (let i = 0; i < numValues; i++) {
        if (nullSet?.has(i)) continue;
        const v = view.getInt32(i * 4, true);
        if (v < min) min = v;
        if (v > max) max = v;
      }
      return min <= max ? { min, max } : null;
    }
    case "int16": {
      let min = 32767, max = -32768;
      for (let i = 0; i < numValues; i++) {
        if (nullSet?.has(i)) continue;
        const v = view.getInt16(i * 2, true);
        if (v < min) min = v;
        if (v > max) max = v;
      }
      return min <= max ? { min, max } : null;
    }
    case "int8": {
      let min = 127, max = -128;
      for (let i = 0; i < numValues; i++) {
        if (nullSet?.has(i)) continue;
        const v = view.getInt8(i);
        if (v < min) min = v;
        if (v > max) max = v;
      }
      return min <= max ? { min, max } : null;
    }
    default:
      return null;
  }
}

/**
 * Decode Lance v2 utf8 data from a buffer containing [offsets_array | padding | string_data].
 * The offsets array is (rowCount) i64 values representing cumulative end positions.
 * String data follows at the position after offsets + alignment padding.
 */
export function decodeLanceV2Utf8(buf: ArrayBuffer, rowCount: number): string[] {
  const view = new DataView(buf);
  const offsetsBytes = rowCount * 8; // rowCount i64 offsets (not rowCount+1)
  if (buf.byteLength < offsetsBytes) return [];

  // Read cumulative end offsets into typed array (avoid per-row BigInt overhead)
  const endOffsets = new Int32Array(rowCount);
  for (let i = 0; i < rowCount; i++) {
    // Read low 32 bits directly — string offsets fit in i32 for practical datasets
    endOffsets[i] = view.getInt32(i * 8, true);
  }

  // Total string data length is the last offset
  const totalStringLen = rowCount > 0 ? endOffsets[rowCount - 1] : 0;
  // String data starts after offsets + padding. Find it by searching backward from end of buffer.
  const dataStart = buf.byteLength - totalStringLen;
  const bytes = new Uint8Array(buf);

  // Decode entire string data block at once (1 TextDecoder call, not N)
  const allStringsDecoded = new TextDecoder().decode(bytes.subarray(dataStart, dataStart + totalStringLen));

  // Fast path: if all strings are ASCII (common), use byte offsets directly as char offsets
  if (allStringsDecoded.length === totalStringLen) {
    const strings = new Array<string>(rowCount);
    let prevEnd = 0;
    for (let i = 0; i < rowCount; i++) {
      const end = endOffsets[i];
      strings[i] = allStringsDecoded.substring(prevEnd, end);
      prevEnd = end;
    }
    return strings;
  }

  // Slow path: multi-byte UTF-8 — must decode per-string to get correct boundaries
  const decoder = new TextDecoder();
  const strings = new Array<string>(rowCount);
  let prevEnd = 0;
  for (let i = 0; i < rowCount; i++) {
    const end = endOffsets[i];
    strings[i] = decoder.decode(bytes.subarray(dataStart + prevEnd, dataStart + end));
    prevEnd = end;
  }
  return strings;
}
