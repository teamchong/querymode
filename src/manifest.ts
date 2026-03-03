import { readVarint, LANCE_MAGIC } from "./footer.js";
import type { DataType, FragmentInfo, ManifestInfo, SchemaField } from "./types.js";

export type { FragmentInfo, ManifestInfo };

/** Map Lance manifest logicalType string to DataType */
export function logicalTypeToDataType(logicalType: string): DataType {
  switch (logicalType) {
    case "int8": return "int8";
    case "int16": return "int16";
    case "int32": return "int32";
    case "int64": return "int64";
    case "uint8": return "uint8";
    case "uint16": return "uint16";
    case "uint32": return "uint32";
    case "uint64": return "uint64";
    case "float16": case "halffloat": return "float16";
    case "float": case "float32": return "float32";
    case "double": case "float64": return "float64";
    case "string": case "utf8": case "large_utf8": return "utf8";
    case "binary": case "large_binary": return "binary";
    case "bool": case "boolean": return "bool";
    case "fixed_size_list": return "fixed_size_list";
    default: return "binary";
  }
}

const decoder = new TextDecoder();

/**
 * Parse a Lance manifest file (_versions/N.manifest).
 *
 * Binary layout:
 *   [0:4]              u32 LE  protobuf length
 *   [4:4+len]          protobuf Manifest message
 *   [4+len:4+len+12]   padding (zeros)
 *   [4+len+12:4+len+16] magic "LANC" (0x434e414c LE)
 */
export function parseManifest(buf: ArrayBuffer): ManifestInfo | null {
  if (buf.byteLength < 20) return null; // 4 (len) + 0 (proto) + 12 (pad) + 4 (magic)

  const view = new DataView(buf);

  const protoLen = view.getUint32(0, true);
  const expectedSize = 4 + protoLen + 12 + 4;
  if (buf.byteLength < expectedSize) return null;

  const magicOffset = 4 + protoLen + 12;
  const magic = view.getUint32(magicOffset, true);
  if (magic !== LANCE_MAGIC) return null;

  const bytes = new Uint8Array(buf, 4, protoLen);
  return parseManifestProto(bytes);
}

/** Parse protobuf Manifest message */
function parseManifestProto(bytes: Uint8Array): ManifestInfo {
  let pos = 0;
  let version = 0;
  const fragments: FragmentInfo[] = [];
  const schema: SchemaField[] = [];

  while (pos < bytes.length) {
    const { value: tag, bytesRead: tagBytes } = readVarint(bytes, pos);
    pos += tagBytes;

    const fieldNumber = tag >> 3;
    const wireType = tag & 0x7;

    if (wireType === 2) {
      const { value: len, bytesRead: lenBytes } = readVarint(bytes, pos);
      pos += lenBytes;

      if (fieldNumber === 1) {
        const field = parseSchemaField(bytes.subarray(pos, pos + len));
        if (field) schema.push(field);
      } else if (fieldNumber === 2) {
        const fragment = parseDataFragment(bytes.subarray(pos, pos + len));
        if (fragment) fragments.push(fragment);
      }

      pos += len;
    } else if (wireType === 0) {
      const { value, bytesRead: valBytes } = readVarint(bytes, pos);
      pos += valBytes;

      if (fieldNumber === 3) version = value;
    } else if (wireType === 5) {
      pos += 4;
    } else if (wireType === 1) {
      pos += 8;
    } else {
      break;
    }
  }

  const totalRows = fragments.reduce((sum, f) => sum + f.physicalRows, 0);
  return { version, fragments, totalRows, schema };
}

/** Parse a schema Field sub-message from the manifest */
function parseSchemaField(bytes: Uint8Array): SchemaField | null {
  let pos = 0;
  let name = "";
  let logicalType = "";
  let id = 0;
  let parentId = 0;
  let nullable = false;

  while (pos < bytes.length) {
    const { value: tag, bytesRead: tagBytes } = readVarint(bytes, pos);
    pos += tagBytes;

    const fieldNumber = tag >> 3;
    const wireType = tag & 0x7;

    if (wireType === 2) {
      const { value: len, bytesRead: lenBytes } = readVarint(bytes, pos);
      pos += lenBytes;

      if (fieldNumber === 2) name = decoder.decode(bytes.subarray(pos, pos + len));
      else if (fieldNumber === 5) logicalType = decoder.decode(bytes.subarray(pos, pos + len));

      pos += len;
    } else if (wireType === 0) {
      const { value, bytesRead: valBytes } = readVarint(bytes, pos);
      pos += valBytes;

      if (fieldNumber === 3) id = value;
      else if (fieldNumber === 4) parentId = value | 0; // signed 32-bit (protobuf int32, not sint32)
      else if (fieldNumber === 6) nullable = value !== 0;
    } else if (wireType === 5) {
      pos += 4;
    } else if (wireType === 1) {
      pos += 8;
    } else {
      break;
    }
  }

  if (!name) return null;
  return { name, logicalType, id, parentId, nullable };
}

/** Parse a DataFragment sub-message */
function parseDataFragment(bytes: Uint8Array): FragmentInfo | null {
  let pos = 0;
  let id = 0;
  let filePath = "";
  let physicalRows = 0;

  while (pos < bytes.length) {
    const { value: tag, bytesRead: tagBytes } = readVarint(bytes, pos);
    pos += tagBytes;

    const fieldNumber = tag >> 3;
    const wireType = tag & 0x7;

    if (wireType === 0) {
      const { value, bytesRead: valBytes } = readVarint(bytes, pos);
      pos += valBytes;

      if (fieldNumber === 1) id = value;
      else if (fieldNumber === 4) physicalRows = value;
    } else if (wireType === 2) {
      const { value: len, bytesRead: lenBytes } = readVarint(bytes, pos);
      pos += lenBytes;

      if (fieldNumber === 2) {
        filePath = parseDataFilePath(bytes.subarray(pos, pos + len));
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

  return { id, filePath, physicalRows };
}

/** Parse a DataFile sub-message, extracting the path string (field 1) */
function parseDataFilePath(bytes: Uint8Array): string {
  let pos = 0;

  while (pos < bytes.length) {
    const { value: tag, bytesRead: tagBytes } = readVarint(bytes, pos);
    pos += tagBytes;

    const fieldNumber = tag >> 3;
    const wireType = tag & 0x7;

    if (wireType === 2) {
      const { value: len, bytesRead: lenBytes } = readVarint(bytes, pos);
      pos += lenBytes;

      if (fieldNumber === 1) {
        return decoder.decode(bytes.subarray(pos, pos + len));
      }

      pos += len;
    } else if (wireType === 0) {
      const { bytesRead: valBytes } = readVarint(bytes, pos);
      pos += valBytes;
    } else if (wireType === 5) {
      pos += 4;
    } else if (wireType === 1) {
      pos += 8;
    } else {
      break;
    }
  }

  return "";
}
