/**
 * CSV/TSV/PSV Reader — streaming delimiter-separated values parser.
 *
 * Supports:
 *   - Comma, tab, and pipe delimiters (auto-detected from first line)
 *   - Quoted fields with escaped quotes ("")
 *   - Type inference from the first N rows (number, boolean, string)
 *   - Proper ColumnMeta + PageInfo generation (single synthetic page per column)
 *   - FragmentSource that yields decoded rows
 */

import { type FormatReader, type DataSource, encodeColumnBuffer } from "../reader.js";
import type { ColumnMeta, DataType, PageInfo, Row } from "../types.js";
import type { FragmentSource } from "../operators.js";

// ---------------------------------------------------------------------------
// CSV Parser Helpers
// ---------------------------------------------------------------------------

/** Detect delimiter from the first line of text. */
function detectDelimiter(firstLine: string): string {
  const candidates = [",", "\t", "|", ";"];
  let best = ",";
  let bestCount = 0;
  for (const d of candidates) {
    const count = countUnquoted(firstLine, d);
    if (count > bestCount) {
      bestCount = count;
      best = d;
    }
  }
  return best;
}

/** Count occurrences of a character outside of quoted fields. */
function countUnquoted(line: string, char: string): number {
  let count = 0;
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    if (line[i] === '"') inQuotes = !inQuotes;
    else if (!inQuotes && line[i] === char) count++;
  }
  return count;
}

/**
 * Parse a single CSV line into fields, respecting quoted fields.
 * Handles fields that contain delimiters, newlines (if pre-joined), and escaped quotes ("").
 * Uses index tracking (slice) instead of per-character string concatenation for O(n) perf.
 */
function parseLine(line: string, delimiter: string): string[] {
  const fields: string[] = [];
  let i = 0;

  while (i < line.length) {
    if (line[i] === '"') {
      // Quoted field — scan for closing quote
      i++; // skip opening quote
      const qStart = i;
      let hasEscape = false;
      while (i < line.length) {
        if (line[i] === '"') {
          if (i + 1 < line.length && line[i + 1] === '"') {
            hasEscape = true;
            i += 2;
          } else {
            break;
          }
        } else {
          i++;
        }
      }
      fields.push(hasEscape ? line.slice(qStart, i).replace(/""/g, '"') : line.slice(qStart, i));
      if (i < line.length) i++; // skip closing quote
      if (i < line.length && line[i] === delimiter) {
        i++; // skip delimiter
        // Trailing delimiter means there's an empty field after it
        if (i === line.length) fields.push("");
      }
    } else {
      // Unquoted field — scan for delimiter
      const start = i;
      while (i < line.length && line[i] !== delimiter) i++;
      fields.push(line.slice(start, i));
      if (i < line.length) {
        i++; // skip delimiter
        // Trailing delimiter means there's an empty field after it
        if (i === line.length) fields.push("");
      }
    }
  }
  if (line.length === 0) fields.push("");
  return fields;
}

/**
 * Split a CSV text into lines, handling quoted fields that contain newlines.
 * Returns an array of complete logical lines (joined across physical newlines
 * when inside quotes). Uses index tracking for O(n) perf.
 */
function splitCsvLines(text: string): string[] {
  const lines: string[] = [];
  let start = 0;
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (!inQuotes && (ch === "\n" || ch === "\r")) {
      if (i > start) {
        lines.push(text.slice(start, i));
      }
      // Handle \r\n
      if (ch === "\r" && i + 1 < text.length && text[i + 1] === "\n") {
        i++;
      }
      start = i + 1;
    }
  }
  if (start < text.length) {
    lines.push(text.slice(start));
  }
  return lines;
}

/** Infer the DataType for a column from sample values. */
function inferType(values: string[]): DataType {
  let allBool = true;
  let allNumber = true;
  let allInt = true;
  let samples = 0;

  for (const v of values) {
    const trimmed = v.trim();
    if (trimmed === "" || trimmed.toLowerCase() === "null" || trimmed.toLowerCase() === "na") continue;
    samples++;

    // Boolean check
    const lower = trimmed.toLowerCase();
    if (lower !== "true" && lower !== "false" && lower !== "0" && lower !== "1") {
      allBool = false;
    }

    // Number check
    const num = Number(trimmed);
    if (isNaN(num)) {
      allNumber = false;
      allInt = false;
    } else if (!Number.isInteger(num) || trimmed.includes(".")) {
      allInt = false;
    }
  }

  if (samples === 0) return "utf8";
  if (allBool) return "bool";
  if (allInt) return "int64";
  if (allNumber) return "float64";
  return "utf8";
}

/** Convert a string value to a typed JS value based on DataType. */
function coerceValue(raw: string, dtype: DataType): number | bigint | string | boolean | null {
  const trimmed = raw.trim();
  if (trimmed === "" || trimmed.toLowerCase() === "null" || trimmed.toLowerCase() === "na") {
    return null;
  }
  switch (dtype) {
    case "bool": {
      const lower = trimmed.toLowerCase();
      return lower === "true" || lower === "1";
    }
    case "int64": {
      const n = Number(trimmed);
      if (!Number.isFinite(n)) return null;
      return BigInt(Math.trunc(n));
    }
    case "float64":
      return Number(trimmed);
    default:
      return trimmed;
  }
}

// ---------------------------------------------------------------------------
// Internal: fully parsed CSV representation
// ---------------------------------------------------------------------------

interface ParsedCsv {
  headers: string[];
  types: DataType[];
  /** Column-oriented data: columns[colIdx][rowIdx] */
  columns: (number | bigint | string | boolean | null)[][];
  rowCount: number;
}

/** Parse an entire CSV buffer into a column-oriented representation. */
export function parseCsvFull(text: string): ParsedCsv {
  const lines = splitCsvLines(text);
  if (lines.length === 0) {
    return { headers: [], types: [], columns: [], rowCount: 0 };
  }

  const delimiter = detectDelimiter(lines[0]);
  const headers = parseLine(lines[0], delimiter).map(h => h.trim());
  const numCols = headers.length;

  // Collect raw string values per column for type inference
  const rawCols: string[][] = Array.from({ length: numCols }, () => []);
  for (let i = 1; i < lines.length; i++) {
    const fields = parseLine(lines[i], delimiter);
    for (let c = 0; c < numCols; c++) {
      rawCols[c].push(fields[c] ?? "");
    }
  }

  // Infer types from all rows (we already have the full dataset in memory)
  const types: DataType[] = rawCols.map(col => inferType(col));

  // Convert to typed column-oriented storage
  const columns: (number | bigint | string | boolean | null)[][] = [];
  const rowCount = rawCols[0]?.length ?? 0;
  for (let c = 0; c < numCols; c++) {
    const typedCol: (number | bigint | string | boolean | null)[] = new Array(rowCount);
    for (let r = 0; r < rowCount; r++) {
      typedCol[r] = coerceValue(rawCols[c][r], types[c]);
    }
    columns.push(typedCol);
  }

  return { headers, types, columns, rowCount };
}

// ---------------------------------------------------------------------------
// Build ColumnMeta from parsed CSV
// ---------------------------------------------------------------------------

function buildColumnMeta(parsed: ParsedCsv): ColumnMeta[] {
  const metas: ColumnMeta[] = [];
  for (let c = 0; c < parsed.headers.length; c++) {
    const values = parsed.columns[c];
    let nullCount = 0;
    let minVal: number | bigint | string | undefined;
    let maxVal: number | bigint | string | undefined;

    for (const v of values) {
      if (v === null) { nullCount++; continue; }
      if (typeof v === "boolean") continue; // skip min/max for booleans
      const comparable = v as number | bigint | string;
      if (minVal === undefined || comparable < minVal) minVal = comparable;
      if (maxVal === undefined || comparable > maxVal) maxVal = comparable;
    }

    // Single synthetic page that covers the whole column
    const page: PageInfo = {
      byteOffset: 0n,
      byteLength: 0, // not meaningful for in-memory CSV
      rowCount: parsed.rowCount,
      nullCount,
      minValue: minVal,
      maxValue: maxVal,
    };

    metas.push({
      name: parsed.headers[c],
      dtype: parsed.types[c],
      pages: [page],
      nullCount,
    });
  }
  return metas;
}

// ---------------------------------------------------------------------------
// CsvFragmentSource — serves decoded rows from pre-parsed CSV
// ---------------------------------------------------------------------------

class CsvFragmentSource implements FragmentSource {
  columns: ColumnMeta[];
  private parsed: ParsedCsv;
  private colIndexMap: Map<string, number>;

  constructor(columns: ColumnMeta[], parsed: ParsedCsv) {
    this.columns = columns;
    this.parsed = parsed;
    this.colIndexMap = new Map();
    for (let i = 0; i < parsed.headers.length; i++) {
      this.colIndexMap.set(parsed.headers[i], i);
    }
  }

  /**
   * readPage is called by ScanOperator for each (column, page) pair.
   * CSV files have a single synthetic page per column. We encode the
   * column's typed values into the binary format that decodePage() expects
   * so the existing pipeline can consume it without changes.
   */
  async readPage(col: ColumnMeta, _page: PageInfo): Promise<ArrayBuffer> {
    const colIdx = this.colIndexMap.get(col.name);
    if (colIdx === undefined) return new ArrayBuffer(0);
    const values = this.parsed.columns[colIdx];
    return encodeColumnBuffer(values, col.dtype);
  }
}

// ---------------------------------------------------------------------------
// CsvReader — implements FormatReader
// ---------------------------------------------------------------------------

export class CsvReader implements FormatReader {
  extensions = [".csv", ".tsv", ".psv", ".txt"];

  /**
   * CSV/TSV files have no magic bytes. Detection works by inspecting the head
   * of the file: if it looks like delimiter-separated text with a header, we
   * claim it.
   */
  canRead(tailBytes: ArrayBuffer, headBytes?: ArrayBuffer): boolean {
    if (!headBytes || headBytes.byteLength === 0) return false;

    const head = new Uint8Array(headBytes);
    // Quick rejection: if the first byte is a binary control character (except
    // BOM U+FEFF which starts with 0xEF in UTF-8), it is not CSV.
    const first = head[0];
    if (first === 0x00 || first === 0x01) return false;
    // Arrow IPC magic "ARROW1"
    if (head.byteLength >= 6 && head[0] === 0x41 && head[1] === 0x52 &&
        head[2] === 0x52 && head[3] === 0x4f && head[4] === 0x57 && head[5] === 0x31) {
      return false;
    }
    // Parquet magic "PAR1"
    if (head.byteLength >= 4 && head[0] === 0x50 && head[1] === 0x41 &&
        head[2] === 0x52 && head[3] === 0x31) {
      return false;
    }
    // JSON array or NDJSON — starts with '[' or '{'
    if (first === 0x5b || first === 0x7b) return false;

    // Decode a chunk of text and check for a delimiter pattern
    const decoder = new TextDecoder("utf-8");
    const sample = decoder.decode(head.slice(0, Math.min(head.length, 2048)));
    const firstLine = sample.split(/\r?\n/)[0] ?? "";
    if (firstLine.length === 0) return false;

    // Must have at least one delimiter on the header line
    const delim = detectDelimiter(firstLine);
    return countUnquoted(firstLine, delim) >= 1;
  }

  async readMeta(source: DataSource): Promise<{ columns: ColumnMeta[]; totalRows: number }> {
    const buf = await source.readAll();
    const text = new TextDecoder().decode(buf);
    const parsed = parseCsvFull(text);
    const columns = buildColumnMeta(parsed);
    return { columns, totalRows: parsed.rowCount };
  }

  async createFragments(source: DataSource, projected: ColumnMeta[]): Promise<FragmentSource[]> {
    const buf = await source.readAll();
    const text = new TextDecoder().decode(buf);
    const parsed = parseCsvFull(text);
    // Build full column meta so the fragment has the full schema
    const allMeta = buildColumnMeta(parsed);
    // Filter to projected columns
    const projectedNames = new Set(projected.map(c => c.name));
    const filteredMeta = allMeta.filter(c => projectedNames.has(c.name));
    return [new CsvFragmentSource(filteredMeta, parsed)];
  }
}
