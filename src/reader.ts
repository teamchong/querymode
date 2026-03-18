/**
 * Pluggable Data Source Readers — Phase 7.
 *
 * Abstracts file-format detection and reading behind a common interface
 * so QueryMode can ingest CSV, JSON/NDJSON, Arrow IPC, and future formats
 * without hardcoding branches in LocalExecutor.
 */

import type { ColumnMeta, DataType } from "./types.js";
import type { FragmentSource } from "./operators.js";

const textEncoder = new TextEncoder();

// ---------------------------------------------------------------------------
// FormatReader — one per file format (CSV, JSON, Arrow IPC, ...)
// ---------------------------------------------------------------------------

export interface FormatReader {
  /** File extensions this reader handles (e.g., [".csv", ".tsv"]) */
  extensions: string[];
  /** Check magic bytes at the tail of a file to detect format.
   *  For text formats that lack magic bytes, this may also inspect the head. */
  canRead(tailBytes: ArrayBuffer, headBytes?: ArrayBuffer): boolean;
  /** Read metadata (columns, row count) from a data source. */
  readMeta(source: DataSource): Promise<{ columns: ColumnMeta[]; totalRows: number }>;
  /** Create fragment sources for query execution. */
  createFragments(source: DataSource, projected: ColumnMeta[]): Promise<FragmentSource[]>;
}

// ---------------------------------------------------------------------------
// DataSource — uniform byte-level access to a file (local, URL, R2, ...)
// ---------------------------------------------------------------------------

export interface DataSource {
  /** Read a byte range from the data. */
  readRange(offset: number, length: number): Promise<ArrayBuffer>;
  /** Get total file size in bytes. */
  getSize(): Promise<number>;
  /** Read the entire file into memory. */
  readAll(): Promise<ArrayBuffer>;
}

// ---------------------------------------------------------------------------
// ReaderRegistry — ordered list of readers; first match wins
// ---------------------------------------------------------------------------

export class ReaderRegistry {
  private readers: FormatReader[] = [];

  register(reader: FormatReader): void {
    this.readers.push(reader);
  }

  /** Detect the format of a data source by probing tail (and optionally head) bytes. */
  async detect(source: DataSource): Promise<FormatReader | null> {
    const size = await source.getSize();
    const tailSize = Math.min(size, 40);
    const tail = await source.readRange(size - tailSize, tailSize);

    // First pass: binary formats that have magic bytes in the tail
    for (const reader of this.readers) {
      if (reader.canRead(tail)) return reader;
    }

    // Second pass: text formats that need the head (CSV, JSON, NDJSON)
    const headSize = Math.min(size, 4096);
    const head = await source.readRange(0, headSize);
    for (const reader of this.readers) {
      if (reader.canRead(tail, head)) return reader;
    }

    return null;
  }

  /** Look up a reader by file extension. */
  getByExtension(ext: string): FormatReader | null {
    const normalized = ext.startsWith(".") ? ext : `.${ext}`;
    for (const reader of this.readers) {
      if (reader.extensions.includes(normalized)) return reader;
    }
    return null;
  }

  /** Return all registered readers. */
  all(): readonly FormatReader[] {
    return this.readers;
  }
}

// ---------------------------------------------------------------------------
// FileDataSource — DataSource backed by a local file (Node/Bun)
// ---------------------------------------------------------------------------

export class FileDataSource implements DataSource {
  private path: string;
  private _size: number | null = null;

  constructor(filePath: string) {
    this.path = filePath;
  }

  async readRange(offset: number, length: number): Promise<ArrayBuffer> {
    const fs = await import("node:fs/promises");
    const handle = await fs.open(this.path, "r");
    try {
      const buf = Buffer.alloc(length);
      await handle.read(buf, 0, length, offset);
      return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    } finally {
      await handle.close();
    }
  }

  async getSize(): Promise<number> {
    if (this._size !== null) return this._size;
    const fs = await import("node:fs/promises");
    const stat = await fs.stat(this.path);
    this._size = Number(stat.size);
    return this._size;
  }

  async readAll(): Promise<ArrayBuffer> {
    const fs = await import("node:fs/promises");
    const buf = await fs.readFile(this.path);
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }
}

// ---------------------------------------------------------------------------
// UrlDataSource — DataSource backed by an HTTP(S) URL
// ---------------------------------------------------------------------------

export class UrlDataSource implements DataSource {
  private url: string;
  private _size: number | null = null;

  constructor(url: string) {
    this.url = url;
  }

  async readRange(offset: number, length: number): Promise<ArrayBuffer> {
    const end = offset + length - 1;
    const resp = await fetch(this.url, {
      headers: { Range: `bytes=${offset}-${end}` },
    });
    return resp.arrayBuffer();
  }

  async getSize(): Promise<number> {
    if (this._size !== null) return this._size;
    const resp = await fetch(this.url, { method: "HEAD" });
    this._size = Number(resp.headers.get("content-length") ?? 0);
    return this._size;
  }

  async readAll(): Promise<ArrayBuffer> {
    const resp = await fetch(this.url);
    return resp.arrayBuffer();
  }
}

// ---------------------------------------------------------------------------
// encodeColumnBuffer — shared by CSV and JSON readers
// ---------------------------------------------------------------------------

/**
 * Encode typed column values into the binary wire format used by decodePage().
 * Used by ingestion readers to produce page data compatible with the scan pipeline.
 */
export function encodeColumnBuffer(
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
      const parts: Uint8Array[] = [];
      let totalLen = 0;
      for (const v of values) {
        const str = v === null ? "" : String(v);
        const encoded = textEncoder.encode(str);
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
