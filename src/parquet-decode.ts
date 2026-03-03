import type { PageEncoding, DataType } from "./types.js";

const textDecoder = new TextDecoder();

// --- Varint ---

function readVarint(bytes: Uint8Array, pos: number): { value: number; bytesRead: number } {
  let value = 0;
  let shift = 0;
  let bytesRead = 0;
  while (pos + bytesRead < bytes.length) {
    const b = bytes[pos + bytesRead];
    value |= (b & 0x7f) << shift;
    bytesRead++;
    if ((b & 0x80) === 0) break;
    shift += 7;
  }
  return { value, bytesRead };
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
        length = 1;
        for (let i = 0; i < extraBytes; i++) {
          length += input[pos++] << (i * 8);
        }
        length += 1;
      }
      output.set(input.subarray(pos, pos + length), outPos);
      pos += length;
      outPos += length;
    } else if (tagType === 1) {
      // Copy with 1-byte offset
      const length = ((tag >> 2) & 0x07) + 4;
      const offset = ((tag & 0xe0) << 3) | input[pos++];
      let srcPos = outPos - offset;
      for (let i = 0; i < length; i++) {
        output[outPos++] = output[srcPos++];
      }
    } else if (tagType === 2) {
      // Copy with 2-byte offset
      const length = ((tag >> 2) & 0x3f) + 1;
      const offset = input[pos] | (input[pos + 1] << 8);
      pos += 2;
      let srcPos = outPos - offset;
      for (let i = 0; i < length; i++) {
        output[outPos++] = output[srcPos++];
      }
    } else {
      // Copy with 4-byte offset
      const length = ((tag >> 2) & 0x3f) + 1;
      const offset = input[pos] | (input[pos + 1] << 8) | (input[pos + 2] << 16) | (input[pos + 3] << 24);
      pos += 4;
      let srcPos = outPos - offset;
      for (let i = 0; i < length; i++) {
        output[outPos++] = output[srcPos++];
      }
    }
  }

  return output;
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
  const mask = (1 << bitWidth) - 1;

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
      let bitPos = 0;
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
        bitPos += bitWidth;
      }

      // Advance past all bit-packed bytes (numGroups * 8 values * bitWidth bits)
      pos += Math.ceil((numGroups * 8 * bitWidth) / 8);
    }
  }

  return { values, bytesRead: pos - offset };
}

// --- Thrift Compact Protocol (minimal inline reader for PageHeader) ---

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

function readZigzagVarint(bytes: Uint8Array, pos: number): { value: number; bytesRead: number } {
  const { value: raw, bytesRead } = readVarint(bytes, pos);
  return { value: (raw >>> 1) ^ -(raw & 1), bytesRead };
}

function parsePageHeader(bytes: Uint8Array, offset: number): PageHeaderInfo | null {
  let pos = offset;
  let pageType = 0;
  let uncompressedSize = 0;
  let compressedSize = 0;
  let numValues = 0;
  let encoding = 0;
  let defLevelsByteLength = 0;
  let repLevelsByteLength = 0;
  let isCompressed = true;
  let lastFieldId = 0;

  // Read top-level PageHeader struct fields
  const result = readThriftStruct(bytes, pos, (fieldId, fieldType, data, dataPos) => {
    switch (fieldId) {
      case 1: // type (i32)
        { const r = readZigzagVarint(data, dataPos); pageType = r.value; return r.bytesRead; }
      case 2: // uncompressed_page_size
        { const r = readZigzagVarint(data, dataPos); uncompressedSize = r.value; return r.bytesRead; }
      case 3: // compressed_page_size
        { const r = readZigzagVarint(data, dataPos); compressedSize = r.value; return r.bytesRead; }
      case 5: // DataPageHeader (struct)
        {
          const sr = readThriftStruct(data, dataPos, (sfid, sft, sd, sdp) => {
            switch (sfid) {
              case 1: { const r = readZigzagVarint(sd, sdp); numValues = r.value; return r.bytesRead; }
              case 2: { const r = readZigzagVarint(sd, sdp); encoding = r.value; return r.bytesRead; }
              case 3: { const r = readZigzagVarint(sd, sdp); return r.bytesRead; } // def encoding
              case 4: { const r = readZigzagVarint(sd, sdp); return r.bytesRead; } // rep encoding
              default: return skipThriftField(sd, sdp, sft);
            }
          });
          return sr;
        }
      case 7: // DictionaryPageHeader (struct)
        {
          const sr = readThriftStruct(data, dataPos, (sfid, sft, sd, sdp) => {
            switch (sfid) {
              case 1: { const r = readZigzagVarint(sd, sdp); numValues = r.value; return r.bytesRead; }
              case 2: { const r = readZigzagVarint(sd, sdp); encoding = r.value; return r.bytesRead; }
              default: return skipThriftField(sd, sdp, sft);
            }
          });
          return sr;
        }
      case 8: // DataPageHeaderV2 (struct)
        {
          const sr = readThriftStruct(data, dataPos, (sfid, sft, sd, sdp) => {
            switch (sfid) {
              case 1: { const r = readZigzagVarint(sd, sdp); numValues = r.value; return r.bytesRead; }
              case 3: { const r = readZigzagVarint(sd, sdp); encoding = r.value; return r.bytesRead; }
              case 4: { const r = readZigzagVarint(sd, sdp); defLevelsByteLength = r.value; return r.bytesRead; }
              case 5: { const r = readZigzagVarint(sd, sdp); repLevelsByteLength = r.value; return r.bytesRead; }
              case 6: { isCompressed = sd[sdp] !== 0; return 1; } // bool
              default: return skipThriftField(sd, sdp, sft);
            }
          });
          return sr;
        }
      default:
        return skipThriftField(data, dataPos, fieldType);
    }
  });

  if (result < 0) return null;

  return {
    type: pageType,
    uncompressedSize,
    compressedSize,
    numValues,
    encoding,
    headerSize: offset + result - offset,
    defLevelsByteLength,
    repLevelsByteLength,
    isCompressed,
  };
}

type FieldHandler = (fieldId: number, fieldType: number, data: Uint8Array, pos: number) => number;

function readThriftStruct(bytes: Uint8Array, offset: number, handler: FieldHandler): number {
  let pos = offset;
  let lastFieldId = 0;

  while (pos < bytes.length) {
    const b = bytes[pos++];
    if (b === 0x00) break; // stop field

    const deltaFieldId = (b >> 4) & 0x0f;
    const fieldType = b & 0x0f;

    let fieldId: number;
    if (deltaFieldId !== 0) {
      fieldId = lastFieldId + deltaFieldId;
    } else {
      // Full field ID follows as i16
      const { value, bytesRead } = readZigzagVarint(bytes, pos);
      fieldId = value;
      pos += bytesRead;
    }
    lastFieldId = fieldId;

    const consumed = handler(fieldId, fieldType, bytes, pos);
    if (consumed < 0) return -1;
    pos += consumed;
  }

  return pos - offset;
}

function skipThriftField(bytes: Uint8Array, pos: number, fieldType: number): number {
  let p = pos;
  switch (fieldType) {
    case 1: // bool true
    case 2: // bool false
      return 0;
    case 3: // i8
      return 1;
    case 4: // i16 (zigzag varint)
    case 5: // i32 (zigzag varint)
    case 6: // i64 (zigzag varint)
      { const r = readVarint(bytes, p); return r.bytesRead; }
    case 7: // double
      return 8;
    case 8: // binary (length-prefixed)
      {
        const { value: len, bytesRead } = readVarint(bytes, p);
        return bytesRead + len;
      }
    case 9: // list
      {
        const header = bytes[p++];
        let size = (header >> 4) & 0x0f;
        const elemType = header & 0x0f;
        if (size === 0x0f) {
          const { value, bytesRead } = readVarint(bytes, p);
          size = value;
          p += bytesRead;
        }
        for (let i = 0; i < size; i++) {
          const skipped = skipThriftField(bytes, p, elemType);
          if (skipped < 0) return -1;
          p += skipped;
        }
        return p - pos;
      }
    case 10: // set (same as list)
      {
        const header = bytes[p++];
        let size = (header >> 4) & 0x0f;
        const elemType = header & 0x0f;
        if (size === 0x0f) {
          const { value, bytesRead } = readVarint(bytes, p);
          size = value;
          p += bytesRead;
        }
        for (let i = 0; i < size; i++) {
          const skipped = skipThriftField(bytes, p, elemType);
          if (skipped < 0) return -1;
          p += skipped;
        }
        return p - pos;
      }
    case 11: // map
      {
        const { value: size, bytesRead } = readVarint(bytes, p);
        p += bytesRead;
        if (size > 0) {
          const types = bytes[p++];
          const keyType = (types >> 4) & 0x0f;
          const valType = types & 0x0f;
          for (let i = 0; i < size; i++) {
            const ks = skipThriftField(bytes, p, keyType);
            if (ks < 0) return -1;
            p += ks;
            const vs = skipThriftField(bytes, p, valType);
            if (vs < 0) return -1;
            p += vs;
          }
        }
        return p - pos;
      }
    case 12: // struct
      {
        const sr = readThriftStruct(bytes, p, (_fid, ftype, data, dpos) => skipThriftField(data, dpos, ftype));
        return sr;
      }
    default:
      return -1;
  }
}

// --- PLAIN Decoding ---

function decodePlainValues(
  bytes: Uint8Array,
  dtype: DataType,
  numValues: number,
): (number | bigint | string | boolean | null)[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values: (number | bigint | string | boolean | null)[] = [];

  switch (dtype) {
    case "int8": {
      const n = Math.min(numValues, bytes.length);
      for (let i = 0; i < n; i++) values.push(new Int8Array(bytes.buffer, bytes.byteOffset + i, 1)[0]);
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
      if (pageEncoding.compression === "SNAPPY") {
        pageData = decompressSnappy(pageData);
      }
      dictionary = decodePlainValues(pageData, dtype, header.numValues);
      pos = pageDataEnd;
      continue;
    }

    if (header.type === 0) {
      // DATA_PAGE (v1)
      if (pageEncoding.compression === "SNAPPY") {
        pageData = decompressSnappy(pageData);
      }

      let dataOffset = 0;

      // Skip repetition levels (assume flat schema, max_rep_level = 0 typically)
      // In v1, def/rep levels are prefixed with 4-byte LE length
      // Definition levels
      if (dataOffset + 4 <= pageData.length) {
        const defLen = new DataView(pageData.buffer, pageData.byteOffset + dataOffset, 4).getUint32(0, true);
        dataOffset += 4 + defLen;
      }

      // Check if dictionary encoding
      const enc = header.encoding;
      if (enc === 8 || enc === 3) {
        // RLE_DICTIONARY or PLAIN_DICTIONARY
        if (dictionary) {
          const bitWidth = pageData[dataOffset];
          dataOffset++;
          if (bitWidth === 0) {
            // All values are index 0
            for (let i = 0; i < header.numValues && values.length < numValues; i++) {
              values.push(dictionary[0] ?? null);
            }
          } else {
            const { values: indices } = decodeRleBitPacked(pageData, dataOffset, bitWidth, header.numValues);
            for (let i = 0; i < indices.length && values.length < numValues; i++) {
              const idx = indices[i];
              values.push(idx < dictionary.length ? dictionary[idx] : null);
            }
          }
        }
      } else {
        // PLAIN encoding
        const remaining = pageData.subarray(dataOffset);
        const decoded = decodePlainValues(remaining, dtype, header.numValues);
        for (let i = 0; i < decoded.length && values.length < numValues; i++) {
          values.push(decoded[i]);
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
      let dataPayload: Uint8Array = pageData.subarray(dataStart);
      if (header.isCompressed && pageEncoding.compression === "SNAPPY") {
        dataPayload = decompressSnappy(dataPayload);
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
            const nonNullCount = defLevels
              ? defLevels.filter(d => d > 0).length
              : header.numValues;
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
        const decoded = decodePlainValues(dataPayload, dtype, header.numValues);
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
