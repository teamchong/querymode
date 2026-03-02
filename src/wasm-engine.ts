/**
 * EdgeQ WASM engine — wraps Zig SIMD compute layer.
 * TS handles orchestration (R2 reads, cache); WASM handles all compute.
 */

import type { Row } from "./types.js";
import type { QueryDescriptor } from "./client.js";

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

const enum WasmColumnType {
  Int64 = 0, Float64 = 1, String = 2, Bool = 3,
  Int32 = 4, Float32 = 5, Vector = 6,
}

export interface WasmExports {
  memory: WebAssembly.Memory;
  wasmAlloc(size: number): number;
  resetHeap(): void;
  openFile(dataPtr: number, len: number): number;
  closeFile(): void;
  getNumColumns(): number;
  getRowCount(colIdx: number): bigint;
  registerTableFragment(namePtr: number, nameLen: number, dataPtr: number, dataLen: number): number;
  registerTableSimpleBinary(namePtr: number, nameLen: number, dataPtr: number, dataLen: number): number;
  hasTable(namePtr: number, nameLen: number): number;
  clearTable(namePtr: number, nameLen: number): void;
  clearTables(): void;
  getSqlInputBuffer(): number;
  setSqlInputLength(len: number): void;
  executeSql(): number;
  getResultSize(): number;
  resetResult(): void;
  getLastError(): number;
  bufferPoolInit(maxSize: number): void;
  bufferPoolSet(keyPtr: number, keyLen: number, dataPtr: number, dataLen: number): number;
  bufferPoolGet(keyPtr: number, keyLen: number, outDataPtr: number, outSize: number): number;
  bufferPoolHas(keyPtr: number, keyLen: number): number;
  bufferPoolDelete(keyPtr: number, keyLen: number): void;
  bufferPoolClear(): void;
  vectorSearchBuffer(
    vectorsPtr: number, numVectors: number, dim: number,
    queryPtr: number, topK: number,
    outIndices: number, outScores: number,
    normalized: number, startIndex: number,
  ): number;
  batchCosineSimilarity(
    queryPtr: number, vectorsPtr: number, dim: number,
    numVectors: number, outScores: number, normalized: number,
  ): void;
  readFloat32Column(colIdx: number, outPtr: number, maxLen: number): number;
  readFloat64Column(colIdx: number, outPtr: number, maxLen: number): number;
  readInt64Column(colIdx: number, outPtr: number, maxLen: number): number;
  readInt32Column(colIdx: number, outPtr: number, maxLen: number): number;
  filterInt64Column(colIdx: number, op: number, value: bigint, outIndices: number, maxIndices: number): number;
  filterFloat64Column(colIdx: number, op: number, value: number, outIndices: number, maxIndices: number): number;
}

export async function instantiateWasm(wasmModule: WebAssembly.Module): Promise<WasmEngine> {
  const instance = await WebAssembly.instantiate(wasmModule, {
    env: {
      opfs_open: () => 0, opfs_close: () => {}, opfs_read: () => 0,
      opfs_write: () => 0, opfs_size: () => 0, opfs_flush: () => {}, opfs_truncate: () => {},
    },
  });
  const engine = new WasmEngine(instance.exports as unknown as WasmExports);
  engine.exports.bufferPoolInit(64 * 1024 * 1024);
  return engine;
}

export class WasmEngine {
  readonly exports: WasmExports;
  constructor(exports: WasmExports) { this.exports = exports; }

  reset(): void { this.exports.resetResult(); }

  private writeString(str: string): { ptr: number; len: number } {
    const bytes = textEncoder.encode(str);
    const ptr = this.exports.wasmAlloc(bytes.length);
    if (ptr) new Uint8Array(this.exports.memory.buffer, ptr, bytes.length).set(bytes);
    return { ptr, len: bytes.length };
  }

  getLastError(): string | null {
    const errPtr = this.exports.getLastError();
    if (!errPtr) return null;
    const mem = new Uint8Array(this.exports.memory.buffer, errPtr);
    let end = 0;
    while (mem[end] !== 0 && end < 1024) end++;
    return textDecoder.decode(mem.subarray(0, end));
  }

  hasTable(name: string): boolean {
    const { ptr, len } = this.writeString(name);
    return ptr ? this.exports.hasTable(ptr, len) !== 0 : false;
  }

  loadTable(name: string, fileBytes: ArrayBuffer): boolean {
    if (this.hasTable(name)) return true;
    const { ptr: namePtr, len: nameLen } = this.writeString(name);
    if (!namePtr) return false;
    const dataPtr = this.exports.wasmAlloc(fileBytes.byteLength);
    if (!dataPtr) return false;
    new Uint8Array(this.exports.memory.buffer, dataPtr, fileBytes.byteLength).set(new Uint8Array(fileBytes));
    return this.exports.registerTableFragment(namePtr, nameLen, dataPtr, fileBytes.byteLength) === 0;
  }

  clearTable(name: string): void {
    const { ptr, len } = this.writeString(name);
    if (ptr) this.exports.clearTable(ptr, len);
  }

  executeQuery(query: QueryDescriptor): Row[] | null {
    const sqlBytes = textEncoder.encode(queryToSql(query));
    const sqlBufPtr = this.exports.getSqlInputBuffer();
    new Uint8Array(this.exports.memory.buffer, sqlBufPtr, sqlBytes.length).set(sqlBytes);
    this.exports.setSqlInputLength(sqlBytes.length);

    const resultPtr = this.exports.executeSql();
    if (!resultPtr) return null;
    const resultSize = this.exports.getResultSize();
    if (!resultSize) return null;

    const rows = parseWasmResult(this.exports.memory.buffer, resultPtr, resultSize);
    this.exports.resetResult();
    return rows;
  }

  vectorSearchBuffer(
    vectors: Float32Array, numVectors: number, dim: number,
    queryVector: Float32Array, topK: number, normalized = false,
  ): { indices: Uint32Array; scores: Float32Array } {
    const totalBytes = vectors.byteLength + queryVector.byteLength + topK * 8;
    const basePtr = this.exports.wasmAlloc(totalBytes);
    if (!basePtr) return { indices: new Uint32Array(0), scores: new Float32Array(0) };

    const vPtr = basePtr;
    new Uint8Array(this.exports.memory.buffer, vPtr, vectors.byteLength).set(
      new Uint8Array(vectors.buffer, vectors.byteOffset, vectors.byteLength),
    );
    const qPtr = vPtr + vectors.byteLength;
    new Uint8Array(this.exports.memory.buffer, qPtr, queryVector.byteLength).set(
      new Uint8Array(queryVector.buffer, queryVector.byteOffset, queryVector.byteLength),
    );
    const iPtr = qPtr + queryVector.byteLength;
    const sPtr = iPtr + topK * 4;

    const resultK = this.exports.vectorSearchBuffer(
      vPtr / 4, numVectors, dim, qPtr / 4, topK,
      iPtr / 4, sPtr / 4, normalized ? 1 : 0, 0,
    );

    return {
      indices: new Uint32Array(this.exports.memory.buffer.slice(iPtr, iPtr + resultK * 4)),
      scores: new Float32Array(this.exports.memory.buffer.slice(sPtr, sPtr + resultK * 4)),
    };
  }
}

// --- SQL generation ---

function queryToSql(query: QueryDescriptor): string {
  const parts: string[] = [];

  if (query.aggregates?.length) {
    const exprs: string[] = [];
    if (query.groupBy) exprs.push(...query.groupBy.map(quote));
    for (const agg of query.aggregates) {
      const expr = `${agg.fn.toUpperCase()}(${agg.column === "*" ? "*" : quote(agg.column)})`;
      exprs.push(agg.alias ? `${expr} AS ${quote(agg.alias)}` : expr);
    }
    parts.push(`SELECT ${exprs.join(", ")}`);
  } else if (query.projections.length > 0) {
    parts.push(`SELECT ${query.projections.join(", ")}`);
  } else {
    parts.push("SELECT *");
  }

  parts.push(`FROM ${quote(query.table)}`);

  if (query.filters.length > 0) {
    const conditions = query.filters.map(f => {
      if (f.op === "in" && Array.isArray(f.value)) {
        return `${quote(f.column)} IN (${f.value.map(v => typeof v === "string" ? `'${v}'` : String(v)).join(", ")})`;
      }
      const opMap: Record<string, string> = { eq: "=", neq: "!=", gt: ">", gte: ">=", lt: "<", lte: "<=" };
      const val = typeof f.value === "string" ? `'${f.value}'` : String(f.value);
      return `${quote(f.column)} ${opMap[f.op]} ${val}`;
    });
    parts.push(`WHERE ${conditions.join(" AND ")}`);
  }

  if (query.vectorSearch) {
    const vs = query.vectorSearch;
    const vecStr = `[${Array.from(vs.queryVector).join(",")}]`;
    parts.push(`${query.filters.length > 0 ? "AND" : "WHERE"} ${quote(vs.column)} NEAR ${vecStr}`);
  }

  if (query.groupBy?.length) parts.push(`GROUP BY ${query.groupBy.map(quote).join(", ")}`);
  if (query.sortColumn) parts.push(`ORDER BY ${quote(query.sortColumn)} ${query.sortDirection?.toUpperCase() ?? "ASC"}`);
  if (query.limit) parts.push(`LIMIT ${query.limit}`);

  return parts.join(" ");
}

function quote(name: string): string {
  return /^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name) ? name : `"${name}"`;
}

// --- WASM result parser ---

function parseWasmResult(memoryBuffer: ArrayBuffer, ptr: number, size: number): Row[] {
  const buf = new Uint8Array(memoryBuffer, ptr, size);
  const view = new DataView(memoryBuffer, ptr, size);
  if (size < 28) return [];

  const numColumns = view.getUint32(4, true);
  const numRows = view.getUint32(8, true);
  const dataStart = view.getUint32(16, true);
  if (!numColumns || !numRows) return [];

  const columns: { name: string; type: number }[] = [];
  let pos = 28;
  for (let c = 0; c < numColumns; c++) {
    const nameLen = buf[pos++];
    const name = textDecoder.decode(buf.subarray(pos, pos + nameLen)); pos += nameLen;
    columns.push({ name, type: buf[pos++] });
  }

  const rows: Row[] = Array.from({ length: numRows }, () => ({}));
  let dp = dataStart;

  for (const col of columns) {
    for (let r = 0; r < numRows; r++) {
      switch (col.type) {
        case WasmColumnType.Int64:   rows[r][col.name] = view.getBigInt64(dp, true); dp += 8; break;
        case WasmColumnType.Float64: rows[r][col.name] = view.getFloat64(dp, true); dp += 8; break;
        case WasmColumnType.Int32:   rows[r][col.name] = view.getInt32(dp, true); dp += 4; break;
        case WasmColumnType.Float32: rows[r][col.name] = view.getFloat32(dp, true); dp += 4; break;
        case WasmColumnType.Bool:    rows[r][col.name] = buf[dp] !== 0; dp += 1; break;
        case WasmColumnType.String: {
          const len = view.getUint32(dp, true); dp += 4;
          rows[r][col.name] = textDecoder.decode(buf.subarray(dp, dp + len)); dp += len;
          break;
        }
        default: rows[r][col.name] = null; dp += 8; break;
      }
    }
  }

  return rows;
}
