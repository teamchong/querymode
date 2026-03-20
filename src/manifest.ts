import { readVarint, LANCE_MAGIC } from "./footer.js";
import type { DataType, FragmentInfo, ManifestInfo, SchemaField } from "./types.js";

const encoder = new TextEncoder();

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

// ---------------------------------------------------------------------------
// Manifest builder — shared by MasterDO and LocalExecutor
// ---------------------------------------------------------------------------

/** Convert DataType to logicalType string for manifest schema */
function dataTypeToLogicalType(dtype: DataType): string {
  switch (dtype) {
    case "float16": return "halffloat";
    case "float32": return "float";
    case "float64": return "double";
    case "bool": return "boolean";
    default: return dtype;
  }
}

/** Derive SchemaField[] from column names+dtypes (used during append) */
export function schemaFromColumns(
  columns: { name: string; dtype: DataType; hasNull?: boolean }[],
): SchemaField[] {
  return columns.map((col, i) => ({
    name: col.name,
    logicalType: dataTypeToLogicalType(col.dtype),
    id: i + 1,
    parentId: 0,
    nullable: col.hasNull ?? false,
  }));
}

function encodeVarint(value: number): Uint8Array {
  const bytes: number[] = [];
  let v = value >>> 0;
  while (v > 0x7f) {
    bytes.push((v & 0x7f) | 0x80);
    v >>>= 7;
  }
  bytes.push(v & 0x7f);
  return new Uint8Array(bytes);
}

/**
 * Build a Lance manifest binary (protobuf + envelope).
 * Output format: [4-byte protoLen LE] [proto] [12 bytes padding] [LANC magic].
 */
export function buildManifestBinary(
  version: number,
  fragments: { id: number; filePath: string; physicalRows: number }[],
  schema?: SchemaField[],
): ArrayBuffer {
  const parts: Uint8Array[] = [];

  // Write schema fields (field 1, length-delimited) — must come before fragments
  if (schema) {
    for (const field of schema) {
      const fieldParts: Uint8Array[] = [];

      // field 2 (string): name
      const nameBytes = encoder.encode(field.name);
      fieldParts.push(new Uint8Array([0x12])); // tag 2, wire type 2
      fieldParts.push(encodeVarint(nameBytes.length));
      fieldParts.push(nameBytes);

      // field 3 (varint): id
      fieldParts.push(new Uint8Array([0x18])); // tag 3, wire type 0
      fieldParts.push(encodeVarint(field.id));

      // field 4 (varint): parentId
      if (field.parentId !== 0) {
        fieldParts.push(new Uint8Array([0x20])); // tag 4, wire type 0
        fieldParts.push(encodeVarint(field.parentId));
      }

      // field 5 (string): logicalType
      const typeBytes = encoder.encode(field.logicalType);
      fieldParts.push(new Uint8Array([0x2a])); // tag 5, wire type 2
      fieldParts.push(encodeVarint(typeBytes.length));
      fieldParts.push(typeBytes);

      // field 6 (varint): nullable
      if (field.nullable) {
        fieldParts.push(new Uint8Array([0x30, 0x01])); // tag 6, value 1
      }

      let fieldLen = 0;
      for (const p of fieldParts) fieldLen += p.length;
      const fieldBuf = new Uint8Array(fieldLen);
      let fOff = 0;
      for (const p of fieldParts) { fieldBuf.set(p, fOff); fOff += p.length; }

      parts.push(new Uint8Array([0x0a])); // manifest field 1, wire type 2
      parts.push(encodeVarint(fieldLen));
      parts.push(fieldBuf);
    }
  }

  // Write each fragment (field 2, length-delimited)
  for (const frag of fragments) {
    const pathBytes = encoder.encode(frag.filePath);
    const fragParts: Uint8Array[] = [];

    // DataFile sub-message: field 1 (path) length-delimited
    const dataFileParts: Uint8Array[] = [
      new Uint8Array([0x0a]), // tag 1, wire type 2
      encodeVarint(pathBytes.length),
      pathBytes,
    ];
    let dataFileLen = 0;
    for (const p of dataFileParts) dataFileLen += p.length;
    const dataFileBuf = new Uint8Array(dataFileLen);
    let dOff = 0;
    for (const p of dataFileParts) { dataFileBuf.set(p, dOff); dOff += p.length; }

    // Fragment sub-fields
    fragParts.push(new Uint8Array([0x08])); // field 1 varint (id)
    fragParts.push(encodeVarint(frag.id));
    fragParts.push(new Uint8Array([0x12])); // field 2 length-delimited (DataFile)
    fragParts.push(encodeVarint(dataFileLen));
    fragParts.push(dataFileBuf);
    fragParts.push(new Uint8Array([0x20])); // field 4 varint (physicalRows)
    fragParts.push(encodeVarint(frag.physicalRows));

    let fragLen = 0;
    for (const p of fragParts) fragLen += p.length;
    const fragBuf = new Uint8Array(fragLen);
    let fOff = 0;
    for (const p of fragParts) { fragBuf.set(p, fOff); fOff += p.length; }

    parts.push(new Uint8Array([0x12])); // manifest field 2 (DataFragment)
    parts.push(encodeVarint(fragLen));
    parts.push(fragBuf);
  }

  // Version (field 3, varint)
  parts.push(new Uint8Array([0x18]));
  parts.push(encodeVarint(version));

  let protoLen = 0;
  for (const p of parts) protoLen += p.length;

  // Envelope: [4-byte len LE] [proto] [12 zero padding] [LANC magic]
  const totalSize = 4 + protoLen + 12 + 4;
  const buf = new ArrayBuffer(totalSize);
  const view = new DataView(buf);
  const bytes = new Uint8Array(buf);

  view.setUint32(0, protoLen, true);
  let off = 4;
  for (const p of parts) { bytes.set(p, off); off += p.length; }
  // 12 bytes padding already zero
  view.setUint32(4 + protoLen + 12, LANCE_MAGIC, true);

  return buf;
}
