import type { PageEncoding, DataType } from "./types.js";
import type { WasmEngine } from "./wasm-engine.js";
import { readVarint } from "./footer.js";
import { ThriftReader } from "./parquet.js";
import { QueryModeError } from "./errors.js";

const textDecoder = new TextDecoder();

function decompressPage(data: Uint8Array, compression: string | undefined, uncompressedSize: number, wasm: WasmEngine): Uint8Array {
  if (!compression || compression === "UNCOMPRESSED") return data;
  switch (compression) {
    case "SNAPPY": return decompressSnappy(data);
    case "ZSTD": return wasm.decompressZstd(data);
    case "GZIP": return wasm.decompressGzip(data);
    case "LZ4_RAW": return wasm.decompressLz4(data, uncompressedSize || data.length * 4);
    case "LZ4": throw new QueryModeError("INVALID_FORMAT", "Parquet LZ4 (Hadoop frame) codec not supported; use LZ4_RAW");
    default: throw new QueryModeError("INVALID_FORMAT", `Unknown Parquet compression codec: ${compression}`);
  }
}

// --- Snappy decompression ---

export function decompressSnappy(input: Uint8Array): Uint8Array {
  let pos = 0;
  const { value: uncompressedLen, bytesRead } = readVarint(input, pos);
  pos += bytesRead;

  const output = new Uint8Array(uncompressedLen);
  let outPos = 0;

  while (pos < input.length && outPos < uncompressedLen) {
    const tag = input[pos++];
    const tagType = tag & 0x03;

    if (tagType === 0) {
      // Literal
      let length = (tag >> 2) & 0x3f;
      if (length < 60) {
        length += 1;
      } else {
        const extraBytes = length - 59;
        if (pos + extraBytes > input.length) break; // malformed
        length = 0;
        for (let i = 0; i < extraBytes; i++) {
          length |= input[pos++] << (i * 8);
        }
        length = (length >>> 0) + 1; // unsigned + 1
      }
      if (pos + length > input.length || outPos + length > output.length) break;
      output.set(input.subarray(pos, pos + length), outPos);
      pos += length;
      outPos += length;
    } else if (tagType === 1) {
      // Copy with 1-byte offset
      if (pos >= input.length) break;
      const length = ((tag >> 2) & 0x07) + 4;
      const offset = ((tag & 0xe0) << 3) | input[pos++];
      if (offset === 0 || offset > outPos) break; // invalid offset
      let srcPos = outPos - offset;
      for (let i = 0; i < length; i++) {
        output[outPos++] = output[srcPos++];
      }
    } else if (tagType === 2) {
      // Copy with 2-byte offset
      if (pos + 2 > input.length) break;
      const length = ((tag >> 2) & 0x3f) + 1;
      const offset = input[pos] | (input[pos + 1] << 8);
      pos += 2;
      if (offset === 0 || offset > outPos) break; // invalid offset
      let srcPos = outPos - offset;
      for (let i = 0; i < length; i++) {
        output[outPos++] = output[srcPos++];
      }
    } else {
      // Copy with 4-byte offset
      if (pos + 4 > input.length) break;
      const length = ((tag >> 2) & 0x3f) + 1;
      const offset = (input[pos] | (input[pos + 1] << 8) | (input[pos + 2] << 16) | (input[pos + 3] << 24)) >>> 0;
      pos += 4;
      if (offset === 0 || offset > outPos) break; // invalid offset
      let srcPos = outPos - offset;
      for (let i = 0; i < length; i++) {
        output[outPos++] = output[srcPos++];
      }
    }
  }

  if (outPos < uncompressedLen) {
    // Decompression terminated early — return only the valid portion
    return output.subarray(0, outPos);
  }
  return output;
}

/** Bytes per value for fixed-width types (0 for variable-length). */
function bytesPerValue(dtype: DataType): number {
  switch (dtype) {
    case "int8": case "uint8": return 1;
    case "bool": return 0; // bit-packed (1/8 byte per value), treat as variable-length for heuristic
    case "int16": case "uint16": case "float16": return 2;
    case "int32": case "uint32": case "float32": return 4;
    case "int64": case "uint64": case "float64": return 8;
    default: return 0; // variable-length (utf8, binary)
  }
}

/** Read V1 definition levels (4-byte length prefix + RLE-encoded levels). Returns null if no levels present. */
function readV1DefLevels(data: Uint8Array, offset: number, numValues: number): { defLevels: number[] | null; newOffset: number } {
  if (offset + 4 > data.length) return { defLevels: null, newOffset: offset };
  const len = data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) | (data[offset + 3] << 24);
  if (len <= 0 || len >= data.length - offset - 4) return { defLevels: null, newOffset: offset };
  const levelData = data.subarray(offset + 4, offset + 4 + len);
  const { values: defs } = decodeRleBitPacked(levelData, 0, 1, numValues);
  return { defLevels: defs, newOffset: offset + 4 + len };
}

// --- RLE/Bit-Packed Hybrid Decoding ---

function decodeRleBitPacked(
  bytes: Uint8Array,
  offset: number,
  bitWidth: number,
  maxValues: number,
): { values: number[]; bytesRead: number } {
  const values: number[] = [];
  let pos = offset;
  const mask = bitWidth >= 32 ? 0xFFFFFFFF : (1 << bitWidth) - 1;

  while (pos < bytes.length && values.length < maxValues) {
    const { value: header, bytesRead } = readVarint(bytes, pos);
    pos += bytesRead;

    if ((header & 1) === 0) {
      // RLE run
      const count = header >> 1;
      const valueBytes = Math.ceil(bitWidth / 8);
      let val = 0;
      for (let i = 0; i < valueBytes && pos < bytes.length; i++) {
        val |= bytes[pos++] << (i * 8);
      }
      val &= mask;
      const n = Math.min(count, maxValues - values.length);
      for (let i = 0; i < n; i++) {
        values.push(val);
      }
    } else {
      // Bit-packed
      const numGroups = header >> 1;
      const totalValues = numGroups * 8;
      const n = Math.min(totalValues, maxValues - values.length);

      if (bitWidth === 0) {
        // bitWidth 0: all values are 0, no data bytes to consume
        for (let i = 0; i < n; i++) values.push(0);
      } else {
        let byteIdx = pos;
        let currentByte = byteIdx < bytes.length ? bytes[byteIdx] : 0;
        let bitsLeft = 8;

        for (let i = 0; i < n; i++) {
          let val = 0;
          let bitsNeeded = bitWidth;
          let valShift = 0;

          while (bitsNeeded > 0) {
            if (bitsLeft === 0) {
              byteIdx++;
              currentByte = byteIdx < bytes.length ? bytes[byteIdx] : 0;
              bitsLeft = 8;
            }
            const take = Math.min(bitsNeeded, bitsLeft);
            val |= (currentByte & ((1 << take) - 1)) << valShift;
            currentByte >>= take;
            bitsLeft -= take;
            bitsNeeded -= take;
            valShift += take;
          }
          values.push(val & mask);
        }
      }

      // Advance past all bit-packed bytes (numGroups * 8 values * bitWidth bits)
      pos += Math.ceil((numGroups * 8 * bitWidth) / 8);
    }
  }

  return { values, bytesRead: pos - offset };
}

// --- Page Header Parsing (using shared ThriftReader) ---

interface PageHeaderInfo {
  type: number;
  uncompressedSize: number;
  compressedSize: number;
  numValues: number;
  encoding: number;
  headerSize: number;
  defLevelsByteLength: number;
  repLevelsByteLength: number;
  isCompressed: boolean;
}

function parseDataPageHeader(r: ThriftReader): { numValues: number; encoding: number } {
  return r.readStruct(r => {
    let numValues = 0, encoding = 0;
    let f: { fieldId: number; typeId: number } | null;
    while ((f = r.nextField()) !== null) {
      switch (f.fieldId) {
        case 1: numValues = r.readI32(); break;
        case 2: encoding = r.readI32(); break;
        default: r.skip(f.typeId); break;
      }
    }
    return { numValues, encoding };
  });
}

function parseDataPageHeaderV2(r: ThriftReader) {
  return r.readStruct(r => {
    let numValues = 0, encoding = 0, defLevelsByteLength = 0, repLevelsByteLength = 0, isCompressed = true;
    let f: { fieldId: number; typeId: number } | null;
    while ((f = r.nextField()) !== null) {
      switch (f.fieldId) {
        case 1: numValues = r.readI32(); break;           // num_values
        case 2: r.readI32(); break;                       // num_nulls (skip)
        case 3: r.readI32(); break;                       // num_rows (skip)
        case 4: encoding = r.readI32(); break;             // encoding
        case 5: defLevelsByteLength = r.readI32(); break;  // definition_levels_byte_length
        case 6: repLevelsByteLength = r.readI32(); break;  // repetition_levels_byte_length
        case 7: isCompressed = f.typeId === 1; break;      // is_compressed (bool in type nibble)
        default: r.skip(f.typeId); break;
      }
    }
    return { numValues, encoding, defLevelsByteLength, repLevelsByteLength, isCompressed };
  });
}

function parsePageHeader(bytes: Uint8Array, offset: number): PageHeaderInfo | null {
  const r = new ThriftReader(bytes, offset);
  let pageType = 0, uncompressedSize = 0, compressedSize = 0;
  let numValues = 0, encoding = 0, defLevelsByteLength = 0, repLevelsByteLength = 0, isCompressed = true;

  let f: { fieldId: number; typeId: number } | null;
  while ((f = r.nextField()) !== null) {
    switch (f.fieldId) {
      case 1: pageType = r.readI32(); break;
      case 2: uncompressedSize = r.readI32(); break;
      case 3: compressedSize = r.readI32(); break;
      case 5: { // DataPageHeader
        const dph = parseDataPageHeader(r);
        numValues = dph.numValues; encoding = dph.encoding;
        break;
      }
      case 7: { // DictionaryPageHeader
        const dph = parseDataPageHeader(r);
        numValues = dph.numValues; encoding = dph.encoding;
        break;
      }
      case 8: { // DataPageHeaderV2
        const v2 = parseDataPageHeaderV2(r);
        numValues = v2.numValues; encoding = v2.encoding;
        defLevelsByteLength = v2.defLevelsByteLength;
        repLevelsByteLength = v2.repLevelsByteLength;
        isCompressed = v2.isCompressed;
        break;
      }
      default: r.skip(f.typeId); break;
    }
  }

  return {
    type: pageType, uncompressedSize, compressedSize, numValues, encoding,
    headerSize: r.ofs - offset, defLevelsByteLength, repLevelsByteLength, isCompressed,
  };
}

// --- PLAIN Decoding ---

export function decodePlainValues(
  bytes: Uint8Array,
  dtype: DataType,
  numValues: number,
): (number | bigint | string | boolean | null)[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values: (number | bigint | string | boolean | null)[] = [];

  switch (dtype) {
    case "int8": {
      const n = Math.min(numValues, bytes.length);
      for (let i = 0; i < n; i++) values.push(view.getInt8(i));
      break;
    }
    case "uint8": {
      const n = Math.min(numValues, bytes.length);
      for (let i = 0; i < n; i++) values.push(bytes[i]);
      break;
    }
    case "int16": {
      const n = Math.min(numValues, bytes.length >> 1);
      for (let i = 0; i < n; i++) values.push(view.getInt16(i * 2, true));
      break;
    }
    case "uint16": {
      const n = Math.min(numValues, bytes.length >> 1);
      for (let i = 0; i < n; i++) values.push(view.getUint16(i * 2, true));
      break;
    }
    case "int32": {
      const n = Math.min(numValues, bytes.length >> 2);
      for (let i = 0; i < n; i++) values.push(view.getInt32(i * 4, true));
      break;
    }
    case "uint32": {
      const n = Math.min(numValues, bytes.length >> 2);
      for (let i = 0; i < n; i++) values.push(view.getUint32(i * 4, true));
      break;
    }
    case "int64": {
      const n = Math.min(numValues, bytes.length >> 3);
      for (let i = 0; i < n; i++) values.push(view.getBigInt64(i * 8, true));
      break;
    }
    case "uint64": {
      const n = Math.min(numValues, bytes.length >> 3);
      for (let i = 0; i < n; i++) values.push(view.getBigUint64(i * 8, true));
      break;
    }
    case "float32": {
      const n = Math.min(numValues, bytes.length >> 2);
      for (let i = 0; i < n; i++) values.push(view.getFloat32(i * 4, true));
      break;
    }
    case "float64": {
      const n = Math.min(numValues, bytes.length >> 3);
      for (let i = 0; i < n; i++) values.push(view.getFloat64(i * 8, true));
      break;
    }
    case "float16": {
      const n = Math.min(numValues, bytes.length >> 1);
      for (let i = 0; i < n; i++) {
        const h = view.getUint16(i * 2, true);
        const s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
        if (e === 0) values.push((s ? -1 : 1) * 2 ** -14 * (m / 1024));
        else if (e === 31) values.push(m ? NaN : s ? -Infinity : Infinity);
        else values.push((s ? -1 : 1) * 2 ** (e - 15) * (1 + m / 1024));
      }
      break;
    }
    case "bool": {
      let decoded = 0;
      for (let b = 0; b < bytes.length && decoded < numValues; b++) {
        for (let bit = 0; bit < 8 && decoded < numValues; bit++, decoded++) {
          values.push(((bytes[b] >> bit) & 1) === 1);
        }
      }
      break;
    }
    case "utf8":
    case "binary": {
      let pos = 0;
      while (values.length < numValues && pos + 4 <= bytes.length) {
        const len = view.getUint32(pos, true);
        pos += 4;
        if (pos + len > bytes.length) break;
        if (dtype === "utf8") {
          values.push(textDecoder.decode(bytes.subarray(pos, pos + len)));
        } else {
          let hex = "";
          for (let i = 0; i < len; i++) hex += bytes[pos + i].toString(16).padStart(2, "0");
          values.push(hex);
        }
        pos += len;
      }
      break;
    }
    case "fixed_size_list": {
      // Should not appear as a Parquet primitive type; treat as float32 array
      const n = Math.min(numValues, bytes.length >> 2);
      for (let i = 0; i < n; i++) values.push(view.getFloat32(i * 4, true));
      break;
    }
  }

  return values;
}

// --- Top-Level Column Chunk Decoder ---

export function decodeParquetColumnChunk(
  chunkBuf: ArrayBuffer,
  pageEncoding: PageEncoding,
  dtype: DataType,
  numValues: number,
  wasm: WasmEngine,
): (number | bigint | string | boolean | null)[] {
  const bytes = new Uint8Array(chunkBuf);
  const values: (number | bigint | string | boolean | null)[] = [];
  let dictionary: (number | bigint | string | boolean | null)[] | null = null;
  let pos = 0;

  while (pos < bytes.length && values.length < numValues) {
    const header = parsePageHeader(bytes, pos);
    if (!header || header.compressedSize <= 0) break;
    pos += header.headerSize;

    const pageDataEnd = pos + header.compressedSize;
    let pageData: Uint8Array = bytes.subarray(pos, pageDataEnd);

    if (header.type === 2) {
      // DICTIONARY_PAGE
      pageData = decompressPage(pageData, pageEncoding.compression, header.uncompressedSize, wasm);
      dictionary = decodePlainValues(pageData, dtype, header.numValues);
      pos = pageDataEnd;
      continue;
    }

    if (header.type === 0) {
      // DATA_PAGE (v1)
      pageData = decompressPage(pageData, pageEncoding.compression, header.uncompressedSize, wasm);

      let dataOffset = 0;
      const enc = header.encoding;

      // V1 DATA_PAGE: detect and decode definition levels for nullable columns.
      // V1 def levels are prefixed with a 4-byte LE length. The length should be
      // small relative to page size (RLE of N boolean values ≈ N/8 bytes).
      let v1DefLevels: number[] | null = null;
      {
        const isDictEnc = enc === 8 || enc === 3;
        let hasLevelSection = false;
        if (dataOffset + 8 <= pageData.length) {
          const prefixLen = pageData[dataOffset] | (pageData[dataOffset + 1] << 8)
            | (pageData[dataOffset + 2] << 16) | (pageData[dataOffset + 3] << 24);
          // Def levels RLE for N values at bitWidth=1 is at most ~N/4 bytes.
          // Reject if prefix looks like data (too large or negative).
          const maxDefBytes = Math.max(64, Math.ceil(header.numValues / 4) + 8);
          if (prefixLen > 0 && prefixLen <= maxDefBytes && prefixLen < pageData.length - dataOffset - 4) {
            if (isDictEnc && dictionary) {
              // For dictionary: also check that first byte after def levels is a valid bitWidth
              const afterDef = dataOffset + 4 + prefixLen;
              if (afterDef < pageData.length) {
                const candidateBW = pageData[afterDef];
                const expectedBW = dictionary.length <= 1 ? 0 : Math.ceil(Math.log2(dictionary.length));
                hasLevelSection = candidateBW === expectedBW;
              }
            } else {
              // For PLAIN: only detect def levels for fixed-width types where we can
              // cross-check remaining data size. Variable-length types (utf8/binary)
              // have no reliable way to distinguish def level prefix from data.
              const bpv = bytesPerValue(dtype);
              if (bpv > 0) {
                const remainingAfterDef = (pageData.length - dataOffset) - 4 - prefixLen;
                hasLevelSection = remainingAfterDef >= 0 && remainingAfterDef <= header.numValues * bpv;
              }
            }
          }
        }
        if (hasLevelSection) {
          const result = readV1DefLevels(pageData, dataOffset, header.numValues);
          v1DefLevels = result.defLevels;
          dataOffset = result.newOffset;
        }
      }
      if (enc === 8 || enc === 3) {
        // RLE_DICTIONARY or PLAIN_DICTIONARY
        if (dictionary) {
          const bitWidth = pageData[dataOffset];
          dataOffset++;
          if (bitWidth === 0) {
            for (let i = 0; i < header.numValues && values.length < numValues; i++) {
              if (v1DefLevels && v1DefLevels[i] === 0) {
                values.push(null);
              } else {
                values.push(dictionary[0] ?? null);
              }
            }
          } else {
            let nonNullCount = header.numValues;
            if (v1DefLevels) {
              nonNullCount = 0;
              for (let di = 0; di < v1DefLevels.length; di++) if (v1DefLevels[di] > 0) nonNullCount++;
            }
            const { values: indices } = decodeRleBitPacked(pageData, dataOffset, bitWidth, nonNullCount);
            let idxPtr = 0;
            for (let i = 0; i < header.numValues && values.length < numValues; i++) {
              if (v1DefLevels && v1DefLevels[i] === 0) {
                values.push(null);
              } else {
                const idx = idxPtr < indices.length ? indices[idxPtr++] : 0;
                values.push(idx < dictionary.length ? dictionary[idx] : null);
              }
            }
          }
        }
      } else {
        // PLAIN encoding
        const plainData = pageData.subarray(dataOffset);
        let nonNullCount = header.numValues;
        if (v1DefLevels) {
          nonNullCount = 0;
          for (let di = 0; di < v1DefLevels.length; di++) if (v1DefLevels[di] > 0) nonNullCount++;
        }
        const decoded = decodePlainValues(plainData, dtype, nonNullCount);
        if (v1DefLevels) {
          let dPtr = 0;
          for (let i = 0; i < header.numValues && values.length < numValues; i++) {
            if (v1DefLevels[i] === 0) {
              values.push(null);
            } else {
              values.push(dPtr < decoded.length ? decoded[dPtr++] : null);
            }
          }
        } else {
          for (let i = 0; i < decoded.length && values.length < numValues; i++) {
            values.push(decoded[i]);
          }
        }
      }

      pos = pageDataEnd;
      continue;
    }

    if (header.type === 3) {
      // DATA_PAGE_V2
      let dataStart = 0;

      // Rep levels (uncompressed in v2)
      const repData = pageData.subarray(dataStart, dataStart + header.repLevelsByteLength);
      dataStart += header.repLevelsByteLength;

      // Def levels (uncompressed in v2)
      const defData = pageData.subarray(dataStart, dataStart + header.defLevelsByteLength);
      dataStart += header.defLevelsByteLength;

      // Actual data (may be compressed)
      // V2 spec: uncompressed_page_size includes rep/def level bytes which are NOT compressed,
      // so subtract them to get the actual data payload uncompressed size.
      let dataPayload: Uint8Array = pageData.subarray(dataStart);
      if (header.isCompressed) {
        const dataUncompressedSize = header.uncompressedSize - header.repLevelsByteLength - header.defLevelsByteLength;
        dataPayload = decompressPage(dataPayload, pageEncoding.compression, dataUncompressedSize, wasm);
      }

      // Decode def levels to find nulls
      let defLevels: number[] | null = null;
      if (header.defLevelsByteLength > 0) {
        const { values: defs } = decodeRleBitPacked(defData, 0, 1, header.numValues);
        defLevels = defs;
      }

      const enc = header.encoding;
      if (enc === 8 || enc === 3) {
        // RLE_DICTIONARY
        if (dictionary) {
          const bitWidth = dataPayload[0];
          let payloadOffset = 1;
          if (bitWidth === 0) {
            for (let i = 0; i < header.numValues && values.length < numValues; i++) {
              if (defLevels && defLevels[i] === 0) {
                values.push(null);
              } else {
                values.push(dictionary[0] ?? null);
              }
            }
          } else {
            let nonNullCount = header.numValues;
            if (defLevels) {
              nonNullCount = 0;
              for (let di = 0; di < defLevels.length; di++) if (defLevels[di] > 0) nonNullCount++;
            }
            const { values: indices } = decodeRleBitPacked(dataPayload, payloadOffset, bitWidth, nonNullCount);
            let idxPtr = 0;
            for (let i = 0; i < header.numValues && values.length < numValues; i++) {
              if (defLevels && defLevels[i] === 0) {
                values.push(null);
              } else {
                const idx = idxPtr < indices.length ? indices[idxPtr++] : 0;
                values.push(idx < dictionary.length ? dictionary[idx] : null);
              }
            }
          }
        }
      } else {
        // PLAIN encoding
        let plainCount = header.numValues;
        if (defLevels) {
          plainCount = 0;
          for (let di = 0; di < defLevels.length; di++) if (defLevels[di] > 0) plainCount++;
        }
        const decoded = decodePlainValues(dataPayload, dtype, plainCount);
        if (defLevels) {
          let dPtr = 0;
          for (let i = 0; i < header.numValues && values.length < numValues; i++) {
            if (defLevels[i] === 0) {
              values.push(null);
            } else {
              values.push(dPtr < decoded.length ? decoded[dPtr++] : null);
            }
          }
        } else {
          for (let i = 0; i < decoded.length && values.length < numValues; i++) {
            values.push(decoded[i]);
          }
        }
      }

      pos = pageDataEnd;
      continue;
    }

    // Unknown page type — skip
    pos = pageDataEnd;
  }

  return values;
}
