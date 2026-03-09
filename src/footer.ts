import type { Footer, ColumnMeta, PageInfo, DataType } from "./types.js";

const decoder = new TextDecoder();
export const LANCE_MAGIC = 0x434e414c; // "LANC" little-endian
export const FOOTER_SIZE = 40;

/** Parse 40-byte Lance footer from raw bytes */
export function parseFooter(buf: ArrayBuffer): Footer | null {
  if (buf.byteLength < FOOTER_SIZE) return null;

  const view = new DataView(buf);
  const offset = buf.byteLength - FOOTER_SIZE;

  const magic = view.getUint32(offset + 36, true);
  if (magic !== LANCE_MAGIC) return null;

  return {
    columnMetaStart: view.getBigUint64(offset, true),
    columnMetaOffsetsStart: view.getBigUint64(offset + 8, true),
    globalBuffOffsetsStart: view.getBigUint64(offset + 16, true),
    numGlobalBuffers: view.getUint32(offset + 24, true),
    numColumns: view.getUint32(offset + 28, true),
    majorVersion: view.getUint16(offset + 32, true),
    minorVersion: view.getUint16(offset + 34, true),
  };
}

/**
 * Parse column metadata from protobuf bytes.
 * Uses a minimal protobuf decoder for Lance's fixed schema.
 *
 * Wire format per column:
 *   field 1 (string):  column name
 *   field 2 (varint):  data type enum
 *   field 3 (message): page info (repeated)
 *   field 4 (varint):  list dimension (for fixed_size_list)
 */
export function parseColumnMetaFromProtobuf(
  buf: ArrayBuffer,
  numColumns: number,
): ColumnMeta[] {
  const columns: ColumnMeta[] = [];
  let pos = 0;
  const bytes = new Uint8Array(buf);

  for (let col = 0; col < numColumns && pos < buf.byteLength; col++) {
    const { value: msgLen, bytesRead } = readVarint(bytes, pos);
    pos += bytesRead;

    if (pos + msgLen > buf.byteLength) break;

    const msgEnd = pos + msgLen;
    let name = `column_${col}`;
    let dtype: DataType = "int64";
    let listDimension: number | undefined;
    let nullCount = 0;
    const pages: PageInfo[] = [];

    while (pos < msgEnd) {
      const { value: tag, bytesRead: tagBytes } = readVarint(bytes, pos);
      pos += tagBytes;

      const fieldNumber = tag >> 3;
      const wireType = tag & 0x7;

      if (wireType === 2) {
        // Length-delimited (string, bytes, embedded message)
        const { value: len, bytesRead: lenBytes } = readVarint(bytes, pos);
        pos += lenBytes;

        if (fieldNumber === 1) {
          name = decoder.decode(bytes.subarray(pos, pos + len));
        } else if (fieldNumber === 3) {
          const pageInfo = parsePageInfo(bytes.subarray(pos, pos + len));
          if (pageInfo) pages.push(pageInfo);
        }

        pos += len;
      } else if (wireType === 0) {
        // Varint
        const { value, bytesRead: valBytes } = readVarint(bytes, pos);
        pos += valBytes;

        if (fieldNumber === 2) {
          dtype = dataTypeFromEnum(value);
        } else if (fieldNumber === 4) {
          listDimension = value;
        } else if (fieldNumber === 5) {
          nullCount = value;
        }
      } else if (wireType === 5) {
        // 32-bit fixed
        pos += 4;
      } else if (wireType === 1) {
        // 64-bit fixed
        pos += 8;
      } else {
        break;
      }
    }

    pos = msgEnd;

    columns.push({
      name,
      dtype,
      pages,
      nullCount,
      listDimension,
    });
  }

  return columns;
}

/**
 * Parse a PageInfo from protobuf bytes.
 *
 * Fields:
 *   1 (varint): byte offset
 *   2 (varint): byte length
 *   3 (varint): row count
 *   4 (varint/string): min value
 *   5 (varint/string): max value
 */
function parsePageInfo(bytes: Uint8Array): PageInfo | null {
  let pos = 0;
  let byteOffset = 0n;
  let byteLength = 0;
  let rowCount = 0;
  let nullCount = 0;
  let minValue: number | bigint | string | undefined;
  let maxValue: number | bigint | string | undefined;

  while (pos < bytes.length) {
    const { value: tag, bytesRead: tagBytes } = readVarint(bytes, pos);
    pos += tagBytes;

    const fieldNumber = tag >> 3;
    const wireType = tag & 0x7;

    if (wireType === 0) {
      const { value, bytesRead: valBytes } = readVarint(bytes, pos);
      pos += valBytes;

      if (fieldNumber === 1) byteOffset = BigInt(value);
      else if (fieldNumber === 2) byteLength = value;
      else if (fieldNumber === 3) rowCount = value;
      else if (fieldNumber === 4) minValue = value;
      else if (fieldNumber === 5) maxValue = value;
      else if (fieldNumber === 6) nullCount = value;
    } else if (wireType === 2) {
      // Length-delimited (string min/max for utf8 columns)
      const { value: len, bytesRead: lenBytes } = readVarint(bytes, pos);
      pos += lenBytes;

      if (fieldNumber === 4) {
        minValue = decoder.decode(bytes.subarray(pos, pos + len));
      } else if (fieldNumber === 5) {
        maxValue = decoder.decode(bytes.subarray(pos, pos + len));
      }

      pos += len;
    } else if (wireType === 5) {
      pos += 4;
    } else if (wireType === 1) {
      pos += 8;
    } else {
      break;
    }
  }

  return { byteOffset, byteLength, rowCount, nullCount, minValue, maxValue };
}

/** Read a varint from a byte array. Consumes all bytes including 64-bit encodings. */
export function readVarint(bytes: Uint8Array, offset: number): { value: number; bytesRead: number } {
  let result = 0;
  let shift = 0;
  let pos = offset;

  while (pos < bytes.length) {
    const byte = bytes[pos++];
    if (shift < 32) {
      result |= (byte & 0x7f) << shift;
    }
    if ((byte & 0x80) === 0) {
      return { value: result >>> 0, bytesRead: pos - offset };
    }
    shift += 7;
  }

  return { value: result >>> 0, bytesRead: pos - offset };
}

/** Map Lance data type enum to string */
const DATA_TYPE_MAP: Record<number, DataType> = {
  0: "int8", 1: "int16", 2: "int32", 3: "int64",
  4: "uint8", 5: "uint16", 6: "uint32", 7: "uint64",
  8: "float16", 9: "float32", 10: "float64",
  11: "utf8", 12: "binary", 13: "bool", 14: "fixed_size_list",
};
function dataTypeFromEnum(value: number): DataType {
  return DATA_TYPE_MAP[value] ?? "int64";
}
