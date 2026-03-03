import type { TableMeta, ColumnMeta, PageInfo, DataType, PageEncoding } from "./types.js";
import { LANCE_MAGIC } from "./footer.js";

const textDecoder = new TextDecoder();

// Thrift compact protocol wire types
const THRIFT_BOOL_TRUE = 1;
const THRIFT_BOOL_FALSE = 2;
const THRIFT_I8 = 3;
const THRIFT_I16 = 4;
const THRIFT_I32 = 5;
const THRIFT_I64 = 6;
const THRIFT_DOUBLE = 7;
const THRIFT_BINARY = 8;
const THRIFT_LIST = 11;
const THRIFT_STRUCT = 12;

export const PARQUET_MAGIC = 0x31524150; // "PAR1" as u32 LE

/** Detect file format from the last 4+ bytes of a file */
export function detectFormat(tailBytes: ArrayBuffer): "lance" | "parquet" | null {
  if (tailBytes.byteLength < 4) return null;
  const view = new DataView(tailBytes);
  const magic = view.getUint32(tailBytes.byteLength - 4, true);
  if (magic === PARQUET_MAGIC) return "parquet";
  if (magic === LANCE_MAGIC) return "lance";
  return null;
}

/** Read the 4-byte footer length from the last 8 bytes of a Parquet file */
export function getParquetFooterLength(last8: ArrayBuffer): number | null {
  if (last8.byteLength < 8) return null;
  const view = new DataView(last8);
  const offset = last8.byteLength - 8;
  const magic = view.getUint32(offset + 4, true);
  if (magic !== PARQUET_MAGIC) return null;
  return view.getUint32(offset, true);
}

// --- Thrift Compact Protocol Reader ---

class ThriftReader {
  private bytes: Uint8Array;
  private offset: number;
  private lastFieldId: number;

  constructor(bytes: Uint8Array, offset = 0) {
    this.bytes = bytes;
    this.offset = offset;
    this.lastFieldId = 0;
  }

  get pos(): number {
    return this.offset;
  }

  /** Read next field header. Returns null on struct end (0x00). */
  nextField(): { fieldId: number; typeId: number } | null {
    if (this.offset >= this.bytes.length) return null;
    const byte = this.bytes[this.offset++];
    if (byte === 0x00) return null; // struct end

    const delta = (byte >> 4) & 0x0f;
    const typeId = byte & 0x0f;

    let fieldId: number;
    if (delta === 0) {
      // Full field ID follows as zigzag i16
      fieldId = this.readZigzagVarint();
    } else {
      fieldId = this.lastFieldId + delta;
    }

    this.lastFieldId = fieldId;
    return { fieldId, typeId };
  }

  /** Read a zigzag-encoded varint as i32 */
  readI32(): number {
    return this.readZigzagVarint();
  }

  /** Read a zigzag-encoded varint as i64 (BigInt) */
  readI64(): bigint {
    return this.readZigzagVarint64();
  }

  /** Read a length-prefixed binary field */
  readBinary(): Uint8Array {
    const len = this.readVarint();
    const result = this.bytes.subarray(this.offset, this.offset + len);
    this.offset += len;
    return result;
  }

  /** Read a length-prefixed string */
  readString(): string {
    return textDecoder.decode(this.readBinary());
  }

  /** Read an unsigned varint */
  readVarint(): number {
    let result = 0;
    let shift = 0;
    while (this.offset < this.bytes.length) {
      const byte = this.bytes[this.offset++];
      result |= (byte & 0x7f) << shift;
      if ((byte & 0x80) === 0) return result >>> 0;
      shift += 7;
      if (shift >= 35) return result >>> 0;
    }
    return result >>> 0;
  }

  /** Skip a field of the given Thrift compact type */
  skip(typeId: number): void {
    switch (typeId) {
      case THRIFT_BOOL_TRUE:
      case THRIFT_BOOL_FALSE:
        break;
      case THRIFT_I8:
        this.offset += 1;
        break;
      case THRIFT_I16:
      case THRIFT_I32:
        this.readVarint(); // consume zigzag varint
        break;
      case THRIFT_I64:
        this.readVarint64(); // consume zigzag varint (64-bit)
        break;
      case THRIFT_DOUBLE:
        this.offset += 8;
        break;
      case THRIFT_BINARY:
        this.readBinary(); // consume length + bytes
        break;
      case THRIFT_LIST: {
        const header = this.bytes[this.offset++];
        let size = (header >> 4) & 0x0f;
        const elemType = header & 0x0f;
        if (size === 0x0f) {
          size = this.readVarint();
        }
        for (let i = 0; i < size; i++) {
          this.skip(elemType);
        }
        break;
      }
      case THRIFT_STRUCT: {
        const savedFieldId = this.lastFieldId;
        this.lastFieldId = 0;
        let field: { fieldId: number; typeId: number } | null;
        while ((field = this.nextField()) !== null) {
          this.skip(field.typeId);
        }
        this.lastFieldId = savedFieldId;
        break;
      }
      default:
        break;
    }
  }

  private readZigzagVarint(): number {
    const n = this.readVarint();
    return (n >>> 1) ^ -(n & 1);
  }

  private readVarint64(): bigint {
    let result = 0n;
    let shift = 0n;
    while (this.offset < this.bytes.length) {
      const byte = this.bytes[this.offset++];
      result |= BigInt(byte & 0x7f) << shift;
      if ((byte & 0x80) === 0) return result;
      shift += 7n;
      if (shift >= 63n) return result;
    }
    return result;
  }

  private readZigzagVarint64(): bigint {
    const n = this.readVarint64();
    return (n >> 1n) ^ -(n & 1n);
  }
}

// --- Parquet metadata interfaces ---

interface ParquetSchemaElement {
  name: string;
  numChildren?: number;
  type?: number;
  convertedType?: number;
  typeLength?: number;
}

interface ParquetStatistics {
  min?: Uint8Array;
  max?: Uint8Array;
  nullCount?: number;
  minValue?: Uint8Array;
  maxValue?: Uint8Array;
}

interface ParquetColumnMetaData {
  type: number;
  codec: number;
  numValues: number;
  totalCompressedSize: number;
  totalUncompressedSize: number;
  dataPageOffset: bigint;
  dictionaryPageOffset?: bigint;
  pathInSchema: string[];
  statistics?: ParquetStatistics;
}

interface ParquetRowGroup {
  columns: ParquetColumnMetaData[];
  numRows: number;
}

interface ParquetFileMetaData {
  version: number;
  schema: ParquetSchemaElement[];
  numRows: number;
  rowGroups: ParquetRowGroup[];
}

// --- Thrift struct parsers ---

function parseStatistics(reader: ThriftReader): ParquetStatistics {
  const stats: ParquetStatistics = {};
  const savedFieldId = reader["lastFieldId"];
  reader["lastFieldId"] = 0;

  let field: { fieldId: number; typeId: number } | null;
  while ((field = reader.nextField()) !== null) {
    switch (field.fieldId) {
      case 1: stats.max = reader.readBinary(); break;
      case 2: stats.min = reader.readBinary(); break;
      case 3: stats.nullCount = Number(reader.readI64()); break;
      case 4: reader.readI64(); break; // distinct_count — skip
      case 5: stats.maxValue = reader.readBinary(); break;
      case 6: stats.minValue = reader.readBinary(); break;
      default: reader.skip(field.typeId); break;
    }
  }

  reader["lastFieldId"] = savedFieldId;
  return stats;
}

function parseColumnMetaData(reader: ThriftReader): ParquetColumnMetaData {
  const meta: ParquetColumnMetaData = {
    type: 0,
    codec: 0,
    numValues: 0,
    totalCompressedSize: 0,
    totalUncompressedSize: 0,
    dataPageOffset: 0n,
    pathInSchema: [],
  };

  const savedFieldId = reader["lastFieldId"];
  reader["lastFieldId"] = 0;

  let field: { fieldId: number; typeId: number } | null;
  while ((field = reader.nextField()) !== null) {
    switch (field.fieldId) {
      case 1: meta.type = reader.readI32(); break;
      case 2: {
        // list<i32> encodings — skip
        const header = reader["bytes"][reader["offset"]++];
        let size = (header >> 4) & 0x0f;
        const elemType = header & 0x0f;
        if (size === 0x0f) size = reader.readVarint();
        for (let i = 0; i < size; i++) reader.skip(elemType);
        break;
      }
      case 3: {
        // list<string> path_in_schema
        const header = reader["bytes"][reader["offset"]++];
        let size = (header >> 4) & 0x0f;
        if (size === 0x0f) size = reader.readVarint();
        for (let i = 0; i < size; i++) {
          meta.pathInSchema.push(reader.readString());
        }
        break;
      }
      case 4: meta.codec = reader.readI32(); break;
      case 5: meta.numValues = Number(reader.readI64()); break;
      case 6: meta.totalUncompressedSize = Number(reader.readI64()); break;
      case 7: meta.totalCompressedSize = Number(reader.readI64()); break;
      case 9: meta.dataPageOffset = reader.readI64(); break;
      case 10: {
        const val = reader.readI64();
        // index_page_offset — skip (not used)
        break;
      }
      case 11: meta.dictionaryPageOffset = reader.readI64(); break;
      case 12: meta.statistics = parseStatistics(reader); break;
      default: reader.skip(field.typeId); break;
    }
  }

  reader["lastFieldId"] = savedFieldId;
  return meta;
}

function parseColumnChunk(reader: ThriftReader): ParquetColumnMetaData {
  let colMeta: ParquetColumnMetaData | null = null;

  const savedFieldId = reader["lastFieldId"];
  reader["lastFieldId"] = 0;

  let field: { fieldId: number; typeId: number } | null;
  while ((field = reader.nextField()) !== null) {
    switch (field.fieldId) {
      case 3:
        // embedded ColumnMetaData struct
        colMeta = parseColumnMetaData(reader);
        break;
      default:
        reader.skip(field.typeId);
        break;
    }
  }

  reader["lastFieldId"] = savedFieldId;

  return colMeta ?? {
    type: 0,
    codec: 0,
    numValues: 0,
    totalCompressedSize: 0,
    totalUncompressedSize: 0,
    dataPageOffset: 0n,
    pathInSchema: [],
  };
}

function parseRowGroup(reader: ThriftReader): ParquetRowGroup {
  const rg: ParquetRowGroup = { columns: [], numRows: 0 };

  const savedFieldId = reader["lastFieldId"];
  reader["lastFieldId"] = 0;

  let field: { fieldId: number; typeId: number } | null;
  while ((field = reader.nextField()) !== null) {
    switch (field.fieldId) {
      case 1: {
        // list<ColumnChunk>
        const header = reader["bytes"][reader["offset"]++];
        let size = (header >> 4) & 0x0f;
        const elemType = header & 0x0f;
        if (size === 0x0f) size = reader.readVarint();
        for (let i = 0; i < size; i++) {
          rg.columns.push(parseColumnChunk(reader));
        }
        break;
      }
      case 2: reader.readI64(); break; // total_byte_size — skip
      case 3: rg.numRows = Number(reader.readI64()); break;
      default: reader.skip(field.typeId); break;
    }
  }

  reader["lastFieldId"] = savedFieldId;
  return rg;
}

function parseSchemaElement(reader: ThriftReader): ParquetSchemaElement {
  const elem: ParquetSchemaElement = { name: "" };

  const savedFieldId = reader["lastFieldId"];
  reader["lastFieldId"] = 0;

  let field: { fieldId: number; typeId: number } | null;
  while ((field = reader.nextField()) !== null) {
    switch (field.fieldId) {
      case 1: elem.name = reader.readString(); break;
      case 2: elem.numChildren = reader.readI32(); break;
      case 3: elem.type = reader.readI32(); break;
      case 4: elem.convertedType = reader.readI32(); break;
      case 6: elem.typeLength = reader.readI32(); break;
      default: reader.skip(field.typeId); break;
    }
  }

  reader["lastFieldId"] = savedFieldId;
  return elem;
}

/**
 * Parse a Parquet FileMetaData from Thrift-encoded footer bytes.
 * The input buffer should contain ONLY the Thrift bytes — no length prefix, no PAR1 magic.
 */
export function parseParquetFooter(tailBuf: ArrayBuffer): ParquetFileMetaData | null {
  const bytes = new Uint8Array(tailBuf);
  if (bytes.length === 0) return null;

  const reader = new ThriftReader(bytes);
  const meta: ParquetFileMetaData = {
    version: 0,
    schema: [],
    numRows: 0,
    rowGroups: [],
  };

  let field: { fieldId: number; typeId: number } | null;
  while ((field = reader.nextField()) !== null) {
    switch (field.fieldId) {
      case 1: meta.version = reader.readI32(); break;
      case 2: {
        // list<SchemaElement>
        const header = bytes[reader.pos];
        reader["offset"]++;
        let size = (header >> 4) & 0x0f;
        if (size === 0x0f) size = reader.readVarint();
        for (let i = 0; i < size; i++) {
          meta.schema.push(parseSchemaElement(reader));
        }
        break;
      }
      case 3: meta.numRows = Number(reader.readI64()); break;
      case 4: {
        // list<RowGroup>
        const header = bytes[reader.pos];
        reader["offset"]++;
        let size = (header >> 4) & 0x0f;
        if (size === 0x0f) size = reader.readVarint();
        for (let i = 0; i < size; i++) {
          meta.rowGroups.push(parseRowGroup(reader));
        }
        break;
      }
      default: reader.skip(field.typeId); break;
    }
  }

  return meta;
}

// --- Conversion to TableMeta ---

/** Map Parquet physical type to DataType */
function parquetPhysicalToDataType(physicalType: number, convertedType?: number): DataType {
  switch (physicalType) {
    case 0: return "bool";
    case 1: return "int32";
    case 2: return "int64";
    case 4: return "float32";
    case 5: return "float64";
    case 6: return convertedType === 0 ? "utf8" : "binary";
    case 7: return "binary"; // FIXED_LEN_BYTE_ARRAY
    default: return "binary";
  }
}

/** Interpret raw statistics bytes as a JS value based on physical type */
function interpretStatBytes(
  raw: Uint8Array,
  physicalType: number,
): number | bigint | string | undefined {
  if (raw.length === 0) return undefined;

  switch (physicalType) {
    case 0: // BOOLEAN
      return raw[0] ? 1 : 0;
    case 1: { // INT32
      if (raw.length < 4) return undefined;
      const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
      return view.getInt32(0, true);
    }
    case 2: { // INT64
      if (raw.length < 8) return undefined;
      const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
      const val = view.getBigInt64(0, true);
      // Return as number if it fits safely
      if (val >= -9007199254740991n && val <= 9007199254740991n) return Number(val);
      return val;
    }
    case 4: { // FLOAT
      if (raw.length < 4) return undefined;
      const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
      return view.getFloat32(0, true);
    }
    case 5: { // DOUBLE
      if (raw.length < 8) return undefined;
      const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
      return view.getFloat64(0, true);
    }
    case 6: // BYTE_ARRAY — interpret as utf8 string
      return textDecoder.decode(raw);
    default:
      return undefined;
  }
}

/** Map Parquet codec enum to compression name */
function codecToCompression(codec: number): string {
  switch (codec) {
    case 0: return "UNCOMPRESSED";
    case 1: return "SNAPPY";
    case 2: return "GZIP";
    case 3: return "LZO";
    case 4: return "BROTLI";
    case 5: return "LZ4";
    case 6: return "ZSTD";
    case 7: return "LZ4_RAW";
    default: return "UNCOMPRESSED";
  }
}

/** Convert parsed Parquet metadata to the unified TableMeta format */
export function parquetMetaToTableMeta(
  meta: ParquetFileMetaData,
  r2Key: string,
  fileSize: bigint,
): TableMeta {
  // Build flat leaf column list from schema tree.
  // Schema[0] is root with numChildren=N, followed by child elements.
  // Leaf elements are those without numChildren or numChildren===0.
  const leaves: { name: string; convertedType?: number }[] = [];
  if (meta.schema.length > 0) {
    // Skip root element (index 0), collect leaves from the rest
    for (let i = 1; i < meta.schema.length; i++) {
      const elem = meta.schema[i];
      if (!elem.numChildren || elem.numChildren === 0) {
        leaves.push({ name: elem.name, convertedType: elem.convertedType });
      }
    }
  }

  // Build a map from column name → leaf index for lookup
  const leafIndexByName = new Map<string, number>();
  for (let i = 0; i < leaves.length; i++) {
    leafIndexByName.set(leaves[i].name, i);
  }

  // Build per-column pages from row groups
  const columnPages = new Map<number, PageInfo[]>();
  for (let i = 0; i < leaves.length; i++) {
    columnPages.set(i, []);
  }

  let totalRows = 0;
  const columnTypes = new Map<number, number>(); // leaf index → physical type

  for (const rg of meta.rowGroups) {
    totalRows += rg.numRows;

    for (const col of rg.columns) {
      // Match column to leaf by path_in_schema[0]
      const colName = col.pathInSchema.length > 0 ? col.pathInSchema[0] : "";
      const leafIdx = leafIndexByName.get(colName);
      if (leafIdx === undefined) continue;

      columnTypes.set(leafIdx, col.type);

      const hasDictionary = col.dictionaryPageOffset !== undefined;
      const byteOffset = hasDictionary ? col.dictionaryPageOffset! : col.dataPageOffset;
      const byteLength = col.totalCompressedSize;

      // Build statistics
      let minValue: number | bigint | string | undefined;
      let maxValue: number | bigint | string | undefined;
      let nullCount = 0;

      if (col.statistics) {
        const stats = col.statistics;
        // Prefer newer min_value/max_value over legacy min/max
        const rawMin = stats.minValue ?? stats.min;
        const rawMax = stats.maxValue ?? stats.max;
        if (rawMin) minValue = interpretStatBytes(rawMin, col.type);
        if (rawMax) maxValue = interpretStatBytes(rawMax, col.type);
        if (stats.nullCount !== undefined) nullCount = stats.nullCount;
      }

      // Build encoding info
      const encoding: PageEncoding = {
        encoding: hasDictionary ? "RLE_DICTIONARY" : "PLAIN",
        compression: codecToCompression(col.codec),
      };

      if (hasDictionary) {
        encoding.dictionaryPageOffset = col.dictionaryPageOffset;
        // Dictionary page length = data page offset - dictionary page offset
        encoding.dictionaryPageLength = Number(col.dataPageOffset - col.dictionaryPageOffset!);
      }

      const page: PageInfo = {
        byteOffset,
        byteLength,
        rowCount: rg.numRows,
        nullCount,
        minValue,
        maxValue,
        encoding,
      };

      columnPages.get(leafIdx)!.push(page);
    }
  }

  // Build ColumnMeta array
  const columns: ColumnMeta[] = [];
  for (let i = 0; i < leaves.length; i++) {
    const leaf = leaves[i];
    const physType = columnTypes.get(i) ?? 6; // default BYTE_ARRAY
    const dtype = parquetPhysicalToDataType(physType, leaf.convertedType);
    const pages = columnPages.get(i) ?? [];

    let totalNullCount = 0;
    for (const p of pages) totalNullCount += p.nullCount;

    columns.push({
      name: leaf.name,
      dtype,
      pages,
      nullCount: totalNullCount,
    });
  }

  // Extract table name from r2Key (filename without extension)
  const lastSlash = r2Key.lastIndexOf("/");
  const basename = lastSlash >= 0 ? r2Key.slice(lastSlash + 1) : r2Key;
  const dotIdx = basename.lastIndexOf(".");
  const tableName = dotIdx >= 0 ? basename.slice(0, dotIdx) : basename;

  return {
    name: tableName,
    format: "parquet",
    columns,
    totalRows: totalRows || meta.numRows,
    fileSize,
    r2Key,
    updatedAt: Date.now(),
  };
}
