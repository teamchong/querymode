/**
 * JSON / NDJSON Reader — supports two formats:
 *
 *   1. JSON array:  [{"a":1}, {"a":2}, ...]
 *   2. NDJSON:      {"a":1}\n{"a":2}\n...
 *
 * Detection: first non-whitespace byte '[' = JSON array, '{' = NDJSON.
 * Schema is inferred from the first N objects.
 */

import type { FormatReader, DataSource } from "../reader.js";
import type { ColumnMeta, DataType, PageInfo, Row } from "../types.js";
import type { FragmentSource } from "../operators.js";

/** Number of rows to sample for type inference. */
const TYPE_INFERENCE_ROWS = 256;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Infer DataType from a JS value. */
function inferTypeFromValue(v: unknown): DataType | null {
  if (v === null || v === undefined) return null;
  switch (typeof v) {
    case "boolean": return "bool";
    case "number": return Number.isInteger(v) ? "int64" : "float64";
    case "bigint": return "int64";
    case "string": return "utf8";
    default: return "utf8";
  }
}

/** Merge two inferred types: number wins over integer; string wins over everything. */
function mergeTypes(a: DataType | null, b: DataType | null): DataType {
  if (a === null) return b ?? "utf8";
  if (b === null) return a;
  if (a === b) return a;
  // bool + anything non-bool => utf8
  if (a === "bool" || b === "bool") return "utf8";
  // int64 + float64 => float64
  if ((a === "int64" && b === "float64") || (a === "float64" && b === "int64")) return "float64";
  // Everything else => utf8
  return "utf8";
}

/** Convert a JS value to a typed Row value based on DataType. */
function coerceValue(raw: unknown, dtype: DataType): number | bigint | string | boolean | null {
  if (raw === null || raw === undefined) return null;
  switch (dtype) {
    case "bool":
      return Boolean(raw);
    case "int64":
      if (typeof raw === "bigint") return raw;
      return BigInt(Math.trunc(Number(raw)));
    case "float64":
      return Number(raw);
    case "utf8":
    case "binary":
      return String(raw);
    case "int32":
    case "int16":
    case "int8":
    case "uint8":
    case "uint16":
    case "uint32":
    case "float32":
      return Number(raw);
    case "uint64":
      if (typeof raw === "bigint") return raw;
      return BigInt(Math.trunc(Number(raw)));
    default:
      return String(raw);
  }
}

// ---------------------------------------------------------------------------
// Parse JSON / NDJSON text into objects
// ---------------------------------------------------------------------------

function parseJsonObjects(text: string): Record<string, unknown>[] {
  const trimmed = text.trim();
  if (trimmed.length === 0) return [];

  if (trimmed[0] === "[") {
    // JSON array
    const arr = JSON.parse(trimmed);
    if (!Array.isArray(arr)) return [];
    return arr.filter((item: unknown): item is Record<string, unknown> => typeof item === "object" && item !== null);
  }

  // NDJSON: one JSON object per line
  const lines = trimmed.split(/\r?\n/);
  const objects: Record<string, unknown>[] = [];
  for (const line of lines) {
    const l = line.trim();
    if (l.length === 0) continue;
    try {
      const obj = JSON.parse(l);
      if (typeof obj === "object" && obj !== null && !Array.isArray(obj)) {
        objects.push(obj);
      }
    } catch {
      // Skip malformed lines in NDJSON
    }
  }
  return objects;
}

// ---------------------------------------------------------------------------
// Schema inference
// ---------------------------------------------------------------------------

interface InferredSchema {
  names: string[];
  types: DataType[];
}

function inferSchema(objects: Record<string, unknown>[]): InferredSchema {
  // Collect all unique keys in insertion order
  const keyOrder: string[] = [];
  const keySet = new Set<string>();
  const typeMap = new Map<string, DataType | null>();

  const sampleSize = Math.min(objects.length, TYPE_INFERENCE_ROWS);
  for (let i = 0; i < sampleSize; i++) {
    const obj = objects[i];
    for (const key of Object.keys(obj)) {
      if (!keySet.has(key)) {
        keySet.add(key);
        keyOrder.push(key);
        typeMap.set(key, null);
      }
      const current = typeMap.get(key) ?? null;
      const inferred = inferTypeFromValue(obj[key]);
      typeMap.set(key, mergeTypes(current, inferred));
    }
  }

  return {
    names: keyOrder,
    types: keyOrder.map(k => typeMap.get(k) ?? "utf8"),
  };
}

// ---------------------------------------------------------------------------
// Internal parsed representation
// ---------------------------------------------------------------------------

interface ParsedJson {
  schema: InferredSchema;
  /** Column-oriented data: columns[colIdx][rowIdx] */
  columns: (number | bigint | string | boolean | null)[][];
  rowCount: number;
}

function parseJsonFull(text: string): ParsedJson {
  const objects = parseJsonObjects(text);
  if (objects.length === 0) {
    return { schema: { names: [], types: [] }, columns: [], rowCount: 0 };
  }

  const schema = inferSchema(objects);
  const numCols = schema.names.length;
  const columns: (number | bigint | string | boolean | null)[][] = Array.from(
    { length: numCols },
    () => new Array(objects.length),
  );

  for (let r = 0; r < objects.length; r++) {
    const obj = objects[r];
    for (let c = 0; c < numCols; c++) {
      const raw = obj[schema.names[c]];
      columns[c][r] = coerceValue(raw, schema.types[c]);
    }
  }

  return { schema, columns, rowCount: objects.length };
}

// ---------------------------------------------------------------------------
// Build ColumnMeta
// ---------------------------------------------------------------------------

function buildColumnMeta(parsed: ParsedJson): ColumnMeta[] {
  const metas: ColumnMeta[] = [];
  for (let c = 0; c < parsed.schema.names.length; c++) {
    const values = parsed.columns[c];
    let nullCount = 0;
    let minVal: number | bigint | string | undefined;
    let maxVal: number | bigint | string | undefined;

    for (const v of values) {
      if (v === null) { nullCount++; continue; }
      if (typeof v === "boolean") continue;
      const comparable = v as number | bigint | string;
      if (minVal === undefined || comparable < minVal) minVal = comparable;
      if (maxVal === undefined || comparable > maxVal) maxVal = comparable;
    }

    const page: PageInfo = {
      byteOffset: 0n,
      byteLength: 0,
      rowCount: parsed.rowCount,
      nullCount,
      minValue: minVal,
      maxValue: maxVal,
    };

    metas.push({
      name: parsed.schema.names[c],
      dtype: parsed.schema.types[c],
      pages: [page],
      nullCount,
    });
  }
  return metas;
}

// ---------------------------------------------------------------------------
// JsonFragmentSource
// ---------------------------------------------------------------------------

class JsonFragmentSource implements FragmentSource {
  columns: ColumnMeta[];
  private parsed: ParsedJson;
  private colIndexMap: Map<string, number>;

  constructor(columns: ColumnMeta[], parsed: ParsedJson) {
    this.columns = columns;
    this.parsed = parsed;
    this.colIndexMap = new Map();
    for (let i = 0; i < parsed.schema.names.length; i++) {
      this.colIndexMap.set(parsed.schema.names[i], i);
    }
  }

  async readPage(col: ColumnMeta, _page: PageInfo): Promise<ArrayBuffer> {
    const colIdx = this.colIndexMap.get(col.name);
    if (colIdx === undefined) return new ArrayBuffer(0);
    const values = this.parsed.columns[colIdx];
    return encodeColumnBuffer(values, col.dtype);
  }
}

// ---------------------------------------------------------------------------
// Encode typed values into the binary format expected by decodePage()
// ---------------------------------------------------------------------------

function encodeColumnBuffer(
  values: (number | bigint | string | boolean | null)[],
  dtype: DataType,
): ArrayBuffer {
  switch (dtype) {
    case "bool": {
      const numBytes = Math.ceil(values.length / 8);
      const buf = new Uint8Array(numBytes);
      for (let i = 0; i < values.length; i++) {
        if (values[i] === true || values[i] === 1) {
          buf[i >> 3] |= 1 << (i & 7);
        }
      }
      return buf.buffer;
    }
    case "int8": {
      const arr = new Int8Array(values.length);
      for (let i = 0; i < values.length; i++) arr[i] = Number(values[i] ?? 0);
      return arr.buffer;
    }
    case "uint8": {
      const arr = new Uint8Array(values.length);
      for (let i = 0; i < values.length; i++) arr[i] = Number(values[i] ?? 0);
      return arr.buffer;
    }
    case "int16": {
      const arr = new Int16Array(values.length);
      for (let i = 0; i < values.length; i++) arr[i] = Number(values[i] ?? 0);
      return arr.buffer;
    }
    case "uint16": {
      const arr = new Uint16Array(values.length);
      for (let i = 0; i < values.length; i++) arr[i] = Number(values[i] ?? 0);
      return arr.buffer;
    }
    case "int32": {
      const arr = new Int32Array(values.length);
      for (let i = 0; i < values.length; i++) arr[i] = Number(values[i] ?? 0);
      return arr.buffer;
    }
    case "uint32": {
      const arr = new Uint32Array(values.length);
      for (let i = 0; i < values.length; i++) arr[i] = Number(values[i] ?? 0);
      return arr.buffer;
    }
    case "int64": {
      const arr = new BigInt64Array(values.length);
      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        arr[i] = typeof v === "bigint" ? v : BigInt(Math.trunc(Number(v ?? 0)));
      }
      return arr.buffer;
    }
    case "uint64": {
      const arr = new BigUint64Array(values.length);
      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        arr[i] = typeof v === "bigint" ? BigInt.asUintN(64, v) : BigInt(Math.trunc(Number(v ?? 0)));
      }
      return arr.buffer;
    }
    case "float32": {
      const arr = new Float32Array(values.length);
      for (let i = 0; i < values.length; i++) arr[i] = Number(values[i] ?? 0);
      return arr.buffer;
    }
    case "float64": {
      const arr = new Float64Array(values.length);
      for (let i = 0; i < values.length; i++) arr[i] = Number(values[i] ?? 0);
      return arr.buffer;
    }
    case "utf8":
    case "binary": {
      const encoder = new TextEncoder();
      const parts: Uint8Array[] = [];
      let totalLen = 0;
      for (const v of values) {
        const str = v === null ? "" : String(v);
        const encoded = encoder.encode(str);
        const header = new Uint8Array(4);
        new DataView(header.buffer).setUint32(0, encoded.length, true);
        parts.push(header, encoded);
        totalLen += 4 + encoded.length;
      }
      const buf = new Uint8Array(totalLen);
      let off = 0;
      for (const p of parts) {
        buf.set(p, off);
        off += p.length;
      }
      return buf.buffer;
    }
    default:
      return new ArrayBuffer(0);
  }
}

// ---------------------------------------------------------------------------
// JsonReader — implements FormatReader
// ---------------------------------------------------------------------------

export class JsonReader implements FormatReader {
  extensions = [".json", ".ndjson", ".jsonl"];

  /**
   * Detect JSON/NDJSON by checking if the first non-whitespace byte is
   * '[' (JSON array) or '{' (NDJSON).
   */
  canRead(_tailBytes: ArrayBuffer, headBytes?: ArrayBuffer): boolean {
    if (!headBytes || headBytes.byteLength === 0) return false;
    const head = new Uint8Array(headBytes);

    // Find first non-whitespace byte
    let first = 0;
    for (let i = 0; i < head.length; i++) {
      const b = head[i];
      if (b === 0x20 || b === 0x09 || b === 0x0a || b === 0x0d) continue;
      // Skip BOM (0xEF 0xBB 0xBF)
      if (b === 0xef && i + 2 < head.length && head[i + 1] === 0xbb && head[i + 2] === 0xbf) {
        i += 2;
        continue;
      }
      first = b;
      break;
    }

    // '[' = 0x5B (JSON array), '{' = 0x7B (NDJSON)
    return first === 0x5b || first === 0x7b;
  }

  async readMeta(source: DataSource): Promise<{ columns: ColumnMeta[]; totalRows: number }> {
    const buf = await source.readAll();
    const text = new TextDecoder().decode(buf);
    const parsed = parseJsonFull(text);
    const columns = buildColumnMeta(parsed);
    return { columns, totalRows: parsed.rowCount };
  }

  async createFragments(source: DataSource, projected: ColumnMeta[]): Promise<FragmentSource[]> {
    const buf = await source.readAll();
    const text = new TextDecoder().decode(buf);
    const parsed = parseJsonFull(text);
    const allMeta = buildColumnMeta(parsed);
    const projectedNames = new Set(projected.map(c => c.name));
    const filteredMeta = allMeta.filter(c => projectedNames.has(c.name));
    return [new JsonFragmentSource(filteredMeta, parsed)];
  }
}
