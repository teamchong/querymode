/**
 * Lance v2 format parser — shared between query-do.ts and local-executor.ts.
 *
 * Lance v2 uses a different protobuf structure for column metadata than v1.
 * The standard parseColumnMetaFromProtobuf() returns 0 pages and generic
 * column_N names for v2 files. This module provides correct v2 parsing.
 *
 * Layout of a Lance v2 file footer (last 40 bytes):
 *   [columnMetaStart: u64] [columnMetaOffsetsStart: u64]
 *   [globalBufOffsetsStart: u64] [numGlobalBuffers: u32]
 *   [numColumns: u32] [majorVersion: u16] [minorVersion: u16] [magic: u32]
 *
 * Column metadata protobuf (per column):
 *   field 1 (LEN) = encoding descriptor
 *   field 2 (LEN) = page data sub-message:
 *     field 2 (LEN) = data type hint
 *     field 3 (VARINT) = row count
 */

import type { ColumnMeta, DataType, SchemaField } from "./types.js";
import { logicalTypeToDataType } from "./manifest.js";

export interface LanceV2ColumnInfo {
  name: string;
  dtype: DataType;
  rowCount: number;
  bytesPerValue: number;
}

/**
 * Parse Lance v2 column metadata from a file's raw bytes.
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

  const colInfos: LanceV2ColumnInfo[] = [];
  for (let col = 0; col < numCols; col++) {
    const metaOffset = Number(new DataView(fileData, colMetaOffsetsStart + col * 8, 8).getBigUint64(0, true));
    const metaEnd = col + 1 < numCols
      ? Number(new DataView(fileData, colMetaOffsetsStart + (col + 1) * 8, 8).getBigUint64(0, true))
      : colMetaOffsetsStart;

    // Parse column protobuf — find field 2 (page sub-message)
    let pageBytes: Uint8Array | null = null;
    let pos = metaOffset;
    while (pos < metaEnd) {
      const tag = fileBytes[pos++];
      const fn = tag >> 3, wt = tag & 7;
      if (wt === 2) {
        let len = 0, shift = 0;
        while (pos < metaEnd) { const b = fileBytes[pos++]; len |= (b & 0x7f) << shift; shift += 7; if (!(b & 0x80)) break; }
        if (fn === 2) pageBytes = fileBytes.subarray(pos, pos + len);
        pos += len;
      } else if (wt === 0) { while (pos < metaEnd && fileBytes[pos++] & 0x80); }
      else if (wt === 1) pos += 8;
      else if (wt === 5) pos += 4;
      else break;
    }

    // Parse page sub-message for row count
    let rowCount = 0;
    if (pageBytes) {
      let pp = 0;
      while (pp < pageBytes.length) {
        const tag = pageBytes[pp++];
        const fn = tag >> 3, wt = tag & 7;
        if (wt === 0) {
          let val = 0, shift = 0;
          while (pp < pageBytes.length) { const b = pageBytes[pp++]; val |= (b & 0x7f) << shift; shift += 7; if (!(b & 0x80)) break; }
          if (fn === 3) rowCount = val;
        } else if (wt === 2) {
          let len = 0, shift = 0;
          while (pp < pageBytes.length) { const b = pageBytes[pp++]; len |= (b & 0x7f) << shift; shift += 7; if (!(b & 0x80)) break; }
          pp += len;
        } else if (wt === 1) pp += 8;
        else if (wt === 5) pp += 4;
        else break;
      }
    }

    if (rowCount === 0 && fallbackTotalRows) rowCount = fallbackTotalRows;

    // Resolve name/type from manifest schema
    let name = `column_${col}`;
    let dtype: DataType = "int64";
    if (leafFields[col]) {
      name = leafFields[col].name;
      dtype = logicalTypeToDataType(leafFields[col].logicalType);
    }

    const bpv = (dtype === "int32" || dtype === "float32" || dtype === "uint32") ? 4
      : (dtype === "int8" || dtype === "uint8" || dtype === "bool") ? 1
      : (dtype === "int16" || dtype === "uint16" || dtype === "float16") ? 2
      : 8;
    colInfos.push({ name, dtype, rowCount, bytesPerValue: bpv });
  }

  return colInfos;
}

/**
 * Convert Lance v2 column info to ColumnMeta with page info.
 * Assumes data is stored sequentially from the beginning of the file.
 */
export function lanceV2ToColumnMeta(colInfos: LanceV2ColumnInfo[]): ColumnMeta[] {
  let dataPos = 0;
  const columns: ColumnMeta[] = [];

  for (const col of colInfos) {
    const dataSize = col.rowCount * col.bytesPerValue;
    columns.push({
      name: col.name,
      dtype: col.dtype,
      nullCount: 0,
      pages: [{
        byteOffset: BigInt(dataPos),
        byteLength: dataSize,
        rowCount: col.rowCount,
        nullCount: 0,
      }],
    });
    dataPos += dataSize;
  }

  return columns;
}
