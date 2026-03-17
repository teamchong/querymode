/**
 * QueryMode WASM engine — wraps Zig SIMD compute layer.
 * TS handles orchestration (R2 reads, cache); WASM handles all compute.
 */

import type { Row, DataType, PageInfo, FilterOp } from "./types.js";
import type { QueryDescriptor } from "./client.js";
import { wasmResultToQMCB } from "./columnar.js";

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

const enum WasmColumnType {
  Int64 = 0, Float64 = 1, String = 2, Bool = 3,
  Int32 = 4, Float32 = 5, Vector = 6,
}

export interface WasmExports {
  memory: WebAssembly.Memory;
  alloc(size: number): number;
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
  bufferPoolGetStats(outCount: number, outBytes: number, outMax: number): void;
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

  // Per-column table registration (sql_executor.zig)
  registerTableInt64(namePtr: number, nameLen: number, colPtr: number, colLen: number, dataPtr: number, rowCount: number): number;
  registerTableFloat64(namePtr: number, nameLen: number, colPtr: number, colLen: number, dataPtr: number, rowCount: number): number;
  registerTableFloat32Vector(namePtr: number, nameLen: number, colPtr: number, colLen: number, dataPtr: number, rowCount: number, vectorDim: number): number;
  registerTableString(namePtr: number, nameLen: number, colPtr: number, colLen: number, offsetsPtr: number, lengthsPtr: number, dataPtr: number, dataLen: number, rowCount: number): number;

  // Buffer SIMD filter (aggregates.zig)
  filterFloat64Buffer(dataPtr: number, len: number, op: number, value: number, outPtr: number, maxOut: number): number;
  filterInt32Buffer(dataPtr: number, len: number, op: number, value: number, outPtr: number, maxOut: number): number;
  filterInt64Buffer(dataPtr: number, len: number, op: number, value: bigint, outPtr: number, maxOut: number): number;
  filterFloat64Range(dataPtr: number, len: number, low: number, high: number, outPtr: number, maxOut: number): number;
  filterInt32Range(dataPtr: number, len: number, low: number, high: number, outPtr: number, maxOut: number): number;
  filterInt64Range(dataPtr: number, len: number, low: bigint, high: bigint, outPtr: number, maxOut: number): number;
  filterFloat64NotRange(dataPtr: number, len: number, low: number, high: number, outPtr: number, maxOut: number): number;
  filterInt32NotRange(dataPtr: number, len: number, low: number, high: number, outPtr: number, maxOut: number): number;
  filterInt64NotRange(dataPtr: number, len: number, low: bigint, high: bigint, outPtr: number, maxOut: number): number;
  filterStringLike(dataPtr: number, offsetsPtr: number, count: number, patternPtr: number, patternLen: number, negated: number, outPtr: number, maxOut: number): number;
  intersectIndices(aPtr: number, aLen: number, bPtr: number, bLen: number, outPtr: number, maxOut: number): number;
  unionIndices(aPtr: number, aLen: number, bPtr: number, bLen: number, outPtr: number, maxOut: number): number;

  // Buffer SIMD aggregates (aggregates.zig) — ptr in element units, len = element count
  sumFloat64Buffer(ptr: number, len: number): number;
  minFloat64Buffer(ptr: number, len: number): number;
  maxFloat64Buffer(ptr: number, len: number): number;
  avgFloat64Buffer(ptr: number, len: number): number;
  sumInt32Buffer(ptr: number, len: number): number;
  sumInt64Buffer(ptr: number, len: number): bigint;
  minInt64Buffer(ptr: number, len: number): bigint;
  maxInt64Buffer(ptr: number, len: number): bigint;

  // Indexed aggregates — operate on filtered subsets (aggregates.zig)
  sumFloat64Indexed(dataPtr: number, indicesPtr: number, count: number): number;
  minFloat64Indexed(dataPtr: number, indicesPtr: number, count: number): number;
  maxFloat64Indexed(dataPtr: number, indicesPtr: number, count: number): number;
  sumInt64Indexed(dataPtr: number, indicesPtr: number, count: number): bigint;
  minInt64Indexed(dataPtr: number, indicesPtr: number, count: number): bigint;
  maxInt64Indexed(dataPtr: number, indicesPtr: number, count: number): bigint;

  // Compression (compression.zig)
  zstd_decompress(compressedPtr: number, compressedLen: number, decompressedPtr: number, decompressedCapacity: number): number;
  zstd_get_decompressed_size(compressedPtr: number, compressedLen: number): number;
  gzip_decompress(compressedPtr: number, compressedLen: number, decompressedPtr: number, decompressedCapacity: number): number;
  lz4_block_decompress(compressedPtr: number, compressedLen: number, decompressedPtr: number, decompressedCapacity: number): number;

  // Fragment reader (fragment_reader.zig)
  fragmentLoad(dataPtr: number, len: number): number;
  fragmentGetColumnCount(): number;
  fragmentGetRowCount(): bigint;
  fragmentGetColumnName(idx: number, outPtr: number, maxLen: number): number;
  fragmentGetColumnType(idx: number, outPtr: number, maxLen: number): number;
  fragmentGetColumnVectorDim(idx: number): number;
  fragmentReadInt64(colIdx: number, outPtr: number, maxCount: number): number;
  fragmentReadFloat64(colIdx: number, outPtr: number, maxCount: number): number;
  fragmentReadFloat32(colIdx: number, outPtr: number, maxCount: number): number;
  fragmentReadInt32(colIdx: number, outPtr: number, maxCount: number): number;

  // Fragment writer (lance_writer.zig via dataset_writer.zig)
  writerInit(capacity: number): number;
  writerGetBuffer(): number;
  writerGetOffset(): number;
  writerReset(): void;
  fragmentBegin(capacity: number): number;
  fragmentAddInt64Column(namePtr: number, nameLen: number, values: number, count: number, nullable: number): number;
  fragmentAddInt32Column(namePtr: number, nameLen: number, values: number, count: number, nullable: number): number;
  fragmentAddFloat64Column(namePtr: number, nameLen: number, values: number, count: number, nullable: number): number;
  fragmentAddFloat32Column(namePtr: number, nameLen: number, values: number, count: number, nullable: number): number;
  fragmentAddStringColumn(namePtr: number, nameLen: number, dataPtr: number, dataLen: number, offsetsPtr: number, count: number, nullable: number): number;
  fragmentAddVectorColumn(namePtr: number, nameLen: number, values: number, count: number, dim: number, nullable: number): number;
  fragmentAddBoolColumn(namePtr: number, nameLen: number, packedBits: number, byteCount: number, rowCount: number, nullable: number): number;
  fragmentEnd(): number;

  // Dataset writer (dataset_writer.zig)
  datasetWriterInit(urlPtr: number, urlLen: number): number;
  setCurrentDataPath(pathPtr: number, pathLen: number): number;
  setUUIDSeed(seed: bigint): void;
  parseLatestVersion(dataPtr: number, dataLen: number): bigint;
  formatVersion(version: bigint, outPtr: number, maxLen: number): number;
  casRetry(): number;
  casReset(): void;
  casGetRetryCount(): number;

  // IVF-PQ vector index (ivf_pq.zig)
  ivfPqLoadIndex(dataPtr: number, dataLen: number): number;
  ivfPqSearch(handle: number, queryPtr: number, dim: number, topK: number, nprobe: number, outIndices: number, outScores: number): number;
  ivfPqFree(handle: number): void;
}

export async function instantiateWasm(wasmModule: WebAssembly.Module): Promise<WasmEngine> {
  let mem: { buffer: ArrayBuffer } | undefined;
  const instance = await WebAssembly.instantiate(wasmModule, {
    env: {
      opfs_open: () => 0, opfs_close: () => {}, opfs_read: () => 0,
      opfs_write: () => 0, opfs_size: () => 0, opfs_flush: () => {}, opfs_truncate: () => {},
      js_log: (ptr: number, len: number) => {
        try {
          if (mem) {
            const msg = textDecoder.decode(new Uint8Array(mem.buffer, ptr, len));
            console.log(`[WASM] ${msg}`);
          }
        } catch { /* ignore */ }
      },
    },
  });
  mem = (instance.exports as unknown as WasmExports).memory;
  const engine = new WasmEngine(instance.exports as unknown as WasmExports);
  engine.exports.bufferPoolInit(64 * 1024 * 1024);
  return engine;
}

export class WasmEngine {
  readonly exports: WasmExports;
  constructor(exports: WasmExports) { this.exports = exports; }

  reset(): void { this.exports.resetResult(); }

  /** Decompress ZSTD data using the Zig std.compress.zstd implementation. */
  decompressZstd(compressed: Uint8Array): Uint8Array {
    const inPtr = this.exports.alloc(compressed.length);
    if (!inPtr) throw new Error("WASM OOM allocating zstd input");
    new Uint8Array(this.exports.memory.buffer, inPtr, compressed.length).set(compressed);

    const decompressedSize = this.exports.zstd_get_decompressed_size(inPtr, compressed.length);
    const capacity = decompressedSize || compressed.length * 4; // estimate if unknown
    const outPtr = this.exports.alloc(capacity);
    if (!outPtr) throw new Error("WASM OOM allocating zstd output");

    const written = this.exports.zstd_decompress(inPtr, compressed.length, outPtr, capacity);
    if (written === 0 && compressed.length > 0) throw new Error("zstd decompression failed");
    return new Uint8Array(this.exports.memory.buffer, outPtr, written).slice();
  }

  /** Decompress GZIP data using the Zig std.compress.gzip implementation. Retries with larger buffer if needed. */
  decompressGzip(compressed: Uint8Array): Uint8Array {
    const inPtr = this.exports.alloc(compressed.length);
    if (!inPtr) throw new Error("WASM OOM allocating gzip input");
    new Uint8Array(this.exports.memory.buffer, inPtr, compressed.length).set(compressed);

    // Try increasing capacities: 4x, 16x, 64x (handles high compression ratios)
    for (const multiplier of [4, 16, 64]) {
      const capacity = compressed.length * multiplier;
      const outPtr = this.exports.alloc(capacity);
      if (!outPtr) throw new Error("WASM OOM allocating gzip output");

      const written = this.exports.gzip_decompress(inPtr, compressed.length, outPtr, capacity);
      if (written > 0) return new Uint8Array(this.exports.memory.buffer, outPtr, written).slice();
      // written === 0 && capacity may have been too small — retry with larger buffer
      if (written === 0 && compressed.length === 0) return new Uint8Array(0);
    }
    throw new Error("gzip decompression failed (output exceeds 64x compressed size)");
  }

  /** Decompress LZ4 block data (Parquet hadoop codec). */
  decompressLz4(compressed: Uint8Array, uncompressedSize: number): Uint8Array {
    const inPtr = this.exports.alloc(compressed.length);
    if (!inPtr) throw new Error("WASM OOM allocating lz4 input");
    new Uint8Array(this.exports.memory.buffer, inPtr, compressed.length).set(compressed);

    const outPtr = this.exports.alloc(uncompressedSize);
    if (!outPtr) throw new Error("WASM OOM allocating lz4 output");

    const written = this.exports.lz4_block_decompress(inPtr, compressed.length, outPtr, uncompressedSize);
    if (written === 0 && compressed.length > 0) throw new Error("lz4 decompression failed");
    return new Uint8Array(this.exports.memory.buffer, outPtr, written).slice();
  }

  private writeString(str: string): { ptr: number; len: number } {
    const bytes = textEncoder.encode(str);
    const ptr = this.exports.alloc(bytes.length);
    if (ptr) new Uint8Array(this.exports.memory.buffer, ptr, bytes.length).set(bytes);
    return { ptr, len: bytes.length };
  }

  /** Write two strings and return [tPtr, tLen, cPtr, cLen] for register* calls. */
  writeStringPair(table: string, col: string): [number, number, number, number] {
    const { ptr: tPtr, len: tLen } = this.writeString(table);
    const { ptr: cPtr, len: cLen } = this.writeString(col);
    return [tPtr, tLen, cPtr, cLen];
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
    const dataPtr = this.exports.alloc(fileBytes.byteLength);
    if (!dataPtr) return false;
    new Uint8Array(this.exports.memory.buffer, dataPtr, fileBytes.byteLength).set(new Uint8Array(fileBytes));
    return this.exports.registerTableFragment(namePtr, nameLen, dataPtr, fileBytes.byteLength) === 0;
  }

  clearTable(name: string): void {
    const { ptr, len } = this.writeString(name);
    if (ptr) this.exports.clearTable(ptr, len);
  }

  /**
   * Register multiple columns' raw page buffers in WASM for SQL execution.
   * Batches allocs by writing the table name once and reusing its pointer.
   * Returns false on WASM OOM (caller should fall back to JS path).
   */
  registerColumns(
    table: string,
    columns: { name: string; dtype: DataType; pages: ArrayBuffer[]; pageInfos: PageInfo[]; listDim?: number }[],
  ): boolean {
    if (columns.length === 0) return true;
    const { ptr: tPtr, len: tLen } = this.writeString(table);
    if (!tPtr) return false;
    for (const col of columns) {
      if (!this.registerColumnInner(tPtr, tLen, col.name, col.dtype, col.pages, col.pageInfos, col.listDim)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Register multiple decoded JS columns in WASM for SQL execution.
   * Batches allocs by writing the table name once.
   */
  registerDecodedColumns(
    table: string,
    columns: { name: string; dtype: DataType; values: (number | bigint | string | boolean | null)[] }[],
  ): boolean {
    if (columns.length === 0) return true;
    const { ptr: tPtr, len: tLen } = this.writeString(table);
    if (!tPtr) return false;
    for (const col of columns) {
      if (!this.registerDecodedColumnInner(tPtr, tLen, col.name, col.dtype, col.values)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Register a column's raw page buffers in WASM for SQL execution.
   * Handles null bitmap stripping, type promotion, and utf8 offset extraction.
   * Returns false on WASM OOM (caller should fall back to JS path).
   */
  registerColumn(
    table: string, colName: string, dtype: DataType,
    pages: ArrayBuffer[], pageInfos: PageInfo[], listDim?: number,
  ): boolean {
    const { ptr: tPtr, len: tLen } = this.writeString(table);
    if (!tPtr) return false;
    return this.registerColumnInner(tPtr, tLen, colName, dtype, pages, pageInfos, listDim);
  }

  private registerColumnInner(
    tPtr: number, tLen: number, colName: string, dtype: DataType,
    pages: ArrayBuffer[], pageInfos: PageInfo[], listDim?: number,
  ): boolean {
    const { ptr: cPtr, len: cLen } = this.writeString(colName);
    if (!cPtr) return false;

    // Concatenate page buffers, stripping null bitmaps
    let totalBytes = 0;
    const cleaned: Uint8Array[] = [];
    let totalRows = 0;

    for (let i = 0; i < pages.length; i++) {
      const pi = pageInfos[i];
      const rowCount = pi?.rowCount ?? 0;
      totalRows += rowCount;
      let raw = new Uint8Array(pages[i]);
      if ((pi?.nullCount ?? 0) > 0 && rowCount > 0) {
        const bitmapBytes = Math.ceil(rowCount / 8);
        raw = raw.subarray(bitmapBytes);
      }
      cleaned.push(raw);
      totalBytes += raw.byteLength;
    }

    if (totalBytes === 0 || totalRows === 0) return true;

    const flat = new Uint8Array(totalBytes);
    let off = 0;
    for (const chunk of cleaned) { flat.set(chunk, off); off += chunk.byteLength; }

    // Dispatch by dtype
    switch (dtype) {
      case "int64": {
        const dataPtr = this.exports.alloc(flat.byteLength);
        if (!dataPtr) return false;
        new Uint8Array(this.exports.memory.buffer, dataPtr, flat.byteLength).set(flat);
        this.exports.registerTableInt64(tPtr, tLen, cPtr, cLen, dataPtr, totalRows);
        return true;
      }

      case "float64": {
        const dataPtr = this.exports.alloc(flat.byteLength);
        if (!dataPtr) return false;
        new Uint8Array(this.exports.memory.buffer, dataPtr, flat.byteLength).set(flat);
        this.exports.registerTableFloat64(tPtr, tLen, cPtr, cLen, dataPtr, totalRows);
        return true;
      }

      case "float32": {
        // Promote float32 → float64 (fixed_size_list vectors use dtype "fixed_size_list")
        const src = new Float32Array(flat.buffer, flat.byteOffset, flat.byteLength >> 2);
        const count = Math.min(totalRows, src.length);
        const f64Bytes = count * 8;
        const dataPtr = this.exports.alloc(f64Bytes);
        if (!dataPtr) return false;
        const dst = new Float64Array(this.exports.memory.buffer, dataPtr, count);
        for (let i = 0; i < count; i++) dst[i] = src[i];
        this.exports.registerTableFloat64(tPtr, tLen, cPtr, cLen, dataPtr, count);
        return true;
      }

      case "int32": case "uint32": {
        // Promote to i64
        const src = dtype === "int32"
          ? new Int32Array(flat.buffer, flat.byteOffset, flat.byteLength >> 2)
          : new Uint32Array(flat.buffer, flat.byteOffset, flat.byteLength >> 2);
        const count = Math.min(totalRows, src.length);
        const i64Bytes = count * 8;
        const dataPtr = this.exports.alloc(i64Bytes);
        if (!dataPtr) return false;
        const dst = new BigInt64Array(this.exports.memory.buffer, dataPtr, count);
        for (let i = 0; i < count; i++) dst[i] = BigInt(src[i]);
        this.exports.registerTableInt64(tPtr, tLen, cPtr, cLen, dataPtr, count);
        return true;
      }

      case "int16": case "uint16": case "int8": case "uint8": {
        // Promote to i64
        let src: ArrayLike<number>;
        switch (dtype) {
          case "int8": src = new Int8Array(flat.buffer, flat.byteOffset, flat.byteLength); break;
          case "uint8": src = new Uint8Array(flat.buffer, flat.byteOffset, flat.byteLength); break;
          case "int16": src = new Int16Array(flat.buffer, flat.byteOffset, flat.byteLength >> 1); break;
          case "uint16": src = new Uint16Array(flat.buffer, flat.byteOffset, flat.byteLength >> 1); break;
        }
        const count = Math.min(totalRows, src.length);
        const i64Bytes = count * 8;
        const dataPtr = this.exports.alloc(i64Bytes);
        if (!dataPtr) return false;
        const dst = new BigInt64Array(this.exports.memory.buffer, dataPtr, count);
        for (let i = 0; i < count; i++) dst[i] = BigInt(src[i]);
        this.exports.registerTableInt64(tPtr, tLen, cPtr, cLen, dataPtr, count);
        return true;
      }

      case "uint64": {
        // Cast to signed i64
        const dataPtr = this.exports.alloc(flat.byteLength);
        if (!dataPtr) return false;
        new Uint8Array(this.exports.memory.buffer, dataPtr, flat.byteLength).set(flat);
        this.exports.registerTableInt64(tPtr, tLen, cPtr, cLen, dataPtr, totalRows);
        return true;
      }

      case "bool": {
        // Unpack bitmap to i64 (0/1)
        const i64Bytes = totalRows * 8;
        const dataPtr = this.exports.alloc(i64Bytes);
        if (!dataPtr) return false;
        const dst = new BigInt64Array(this.exports.memory.buffer, dataPtr, totalRows);
        let idx = 0;
        for (let b = 0; b < flat.byteLength && idx < totalRows; b++) {
          for (let bit = 0; bit < 8 && idx < totalRows; bit++, idx++) {
            dst[idx] = BigInt((flat[b] >> bit) & 1);
          }
        }
        this.exports.registerTableInt64(tPtr, tLen, cPtr, cLen, dataPtr, totalRows);
        return true;
      }

      case "fixed_size_list": {
        const dim = listDim ?? 0;
        if (dim === 0) return true;
        const dataPtr = this.exports.alloc(flat.byteLength);
        if (!dataPtr) return false;
        new Uint8Array(this.exports.memory.buffer, dataPtr, flat.byteLength).set(flat);
        this.exports.registerTableFloat32Vector(tPtr, tLen, cPtr, cLen, dataPtr, totalRows, dim);
        return true;
      }

      case "utf8": {
        // Single-pass: build offsets/lengths and copy string data simultaneously
        const offsets = new Uint32Array(totalRows);
        const lengths = new Uint32Array(totalRows);
        const strData = new Uint8Array(flat.byteLength); // upper bound; trimmed at WASM copy
        const view = new DataView(flat.buffer, flat.byteOffset, flat.byteLength);
        let pos = 0;
        let dOff = 0;
        let ri = 0;
        while (pos + 4 <= flat.byteLength && ri < totalRows) {
          const strLen = view.getUint32(pos, true);
          pos += 4;
          offsets[ri] = dOff;
          lengths[ri] = strLen;
          if (strLen > 0 && pos + strLen <= flat.byteLength) {
            strData.set(flat.subarray(pos, pos + strLen), dOff);
          }
          dOff += strLen;
          pos += strLen;
          ri++;
        }
        const dataLen = dOff;

        const offsetsPtr = this.exports.alloc(offsets.byteLength);
        if (!offsetsPtr) return false;
        new Uint32Array(this.exports.memory.buffer, offsetsPtr, ri).set(offsets.subarray(0, ri));

        const lengthsPtr = this.exports.alloc(lengths.byteLength);
        if (!lengthsPtr) return false;
        new Uint32Array(this.exports.memory.buffer, lengthsPtr, ri).set(lengths.subarray(0, ri));

        const dataPtr = this.exports.alloc(dataLen || 1);
        if (!dataPtr) return false;
        if (dataLen > 0) {
          new Uint8Array(this.exports.memory.buffer, dataPtr, dataLen).set(strData);
        }

        this.exports.registerTableString(tPtr, tLen, cPtr, cLen, offsetsPtr, lengthsPtr, dataPtr, dataLen, ri);
        return true;
      }

      default:
        throw new Error(`Unsupported dtype for WASM registration: ${dtype}`);
    }
  }

  /**
   * Register a decoded JS column (from Parquet decode) into WASM for SQL execution.
   * Converts JS arrays to typed arrays and calls the appropriate register* export.
   * Returns false on WASM OOM or unsupported type.
   */
  registerDecodedColumn(
    table: string, colName: string, dtype: DataType,
    values: (number | bigint | string | boolean | null)[],
  ): boolean {
    if (values.length === 0) return true;
    const { ptr: tPtr, len: tLen } = this.writeString(table);
    if (!tPtr) return false;
    return this.registerDecodedColumnInner(tPtr, tLen, colName, dtype, values);
  }

  private registerDecodedColumnInner(
    tPtr: number, tLen: number, colName: string, dtype: DataType,
    values: (number | bigint | string | boolean | null)[],
  ): boolean {
    if (values.length === 0) return true;
    const { ptr: cPtr, len: cLen } = this.writeString(colName);
    if (!cPtr) return false;
    const rowCount = values.length;

    switch (dtype) {
      case "float64": case "float32": {
        const dataPtr = this.exports.alloc(rowCount * 8);
        if (!dataPtr) return false;
        const dst = new Float64Array(this.exports.memory.buffer, dataPtr, rowCount);
        for (let i = 0; i < rowCount; i++) dst[i] = (values[i] as number) ?? 0;
        this.exports.registerTableFloat64(tPtr, tLen, cPtr, cLen, dataPtr, rowCount);
        return true;
      }
      case "int64": {
        const dataPtr = this.exports.alloc(rowCount * 8);
        if (!dataPtr) return false;
        const dst = new BigInt64Array(this.exports.memory.buffer, dataPtr, rowCount);
        for (let i = 0; i < rowCount; i++) dst[i] = BigInt(values[i] as bigint ?? 0n);
        this.exports.registerTableInt64(tPtr, tLen, cPtr, cLen, dataPtr, rowCount);
        return true;
      }
      case "int32": case "int16": case "int8":
      case "uint32": case "uint16": case "uint8": {
        // Promote to i64
        const dataPtr = this.exports.alloc(rowCount * 8);
        if (!dataPtr) return false;
        const dst = new BigInt64Array(this.exports.memory.buffer, dataPtr, rowCount);
        for (let i = 0; i < rowCount; i++) dst[i] = BigInt(Math.trunc((values[i] as number) ?? 0));
        this.exports.registerTableInt64(tPtr, tLen, cPtr, cLen, dataPtr, rowCount);
        return true;
      }
      case "bool": {
        const dataPtr = this.exports.alloc(rowCount * 8);
        if (!dataPtr) return false;
        const dst = new BigInt64Array(this.exports.memory.buffer, dataPtr, rowCount);
        for (let i = 0; i < rowCount; i++) dst[i] = values[i] ? 1n : 0n;
        this.exports.registerTableInt64(tPtr, tLen, cPtr, cLen, dataPtr, rowCount);
        return true;
      }
      case "utf8": {
        // Two passes: measure total string bytes, then pack
        let totalBytes = 0;
        const encoded: Uint8Array[] = [];
        for (let i = 0; i < rowCount; i++) {
          const s = values[i] != null ? String(values[i]) : "";
          const bytes = textEncoder.encode(s);
          encoded.push(bytes);
          totalBytes += bytes.byteLength;
        }
        const offsets = new Uint32Array(rowCount);
        const lengths = new Uint32Array(rowCount);
        const strData = new Uint8Array(totalBytes);
        let off = 0;
        for (let i = 0; i < rowCount; i++) {
          offsets[i] = off;
          lengths[i] = encoded[i].byteLength;
          strData.set(encoded[i], off);
          off += encoded[i].byteLength;
        }
        const offsetsPtr = this.exports.alloc(offsets.byteLength);
        if (!offsetsPtr) return false;
        new Uint32Array(this.exports.memory.buffer, offsetsPtr, rowCount).set(offsets);
        const lengthsPtr = this.exports.alloc(lengths.byteLength);
        if (!lengthsPtr) return false;
        new Uint32Array(this.exports.memory.buffer, lengthsPtr, rowCount).set(lengths);
        const dataPtr = this.exports.alloc(totalBytes || 1);
        if (!dataPtr) return false;
        if (totalBytes > 0) {
          new Uint8Array(this.exports.memory.buffer, dataPtr, totalBytes).set(strData);
        }
        this.exports.registerTableString(
          tPtr, tLen, cPtr, cLen, offsetsPtr, lengthsPtr, dataPtr, totalBytes, rowCount,
        );
        return true;
      }
      default:
        return false; // Unsupported type — caller falls back to JS
    }
  }

  /** Check if a key exists in the WASM buffer pool cache. */
  cacheHas(key: string): boolean {
    const { ptr, len } = this.writeString(key);
    if (!ptr) return false;
    return this.exports.bufferPoolHas(ptr, len) !== 0;
  }

  /** Store data in the WASM buffer pool cache. Returns false on failure (not fatal). */
  cacheSet(key: string, data: ArrayBuffer): boolean {
    const { ptr: keyPtr, len: keyLen } = this.writeString(key);
    if (!keyPtr) return false;
    const dataPtr = this.exports.alloc(data.byteLength);
    if (!dataPtr) return false;
    new Uint8Array(this.exports.memory.buffer, dataPtr, data.byteLength).set(new Uint8Array(data));
    return this.exports.bufferPoolSet(keyPtr, keyLen, dataPtr, data.byteLength) !== 0;
  }

  /** Retrieve data from the WASM buffer pool cache. Returns null on miss.
   *  Uses .slice() to copy data out — survives resetHeap(). */
  cacheGet(key: string): ArrayBuffer | null {
    const { ptr: keyPtr, len: keyLen } = this.writeString(key);
    if (!keyPtr) return null;
    // Allocate two 4-byte out-params: [dataPtr: u32, size: u32]
    const outPtr = this.exports.alloc(8);
    if (!outPtr) return null;
    const found = this.exports.bufferPoolGet(keyPtr, keyLen, outPtr, outPtr + 4);
    if (!found) return null;
    const view = new DataView(this.exports.memory.buffer);
    const dataPtr = view.getUint32(outPtr, true);
    const size = view.getUint32(outPtr + 4, true);
    if (!dataPtr || !size) return null;
    return this.exports.memory.buffer.slice(dataPtr, dataPtr + size);
  }

  /** Clear all entries from the WASM buffer pool cache. */
  cacheClear(): void {
    this.exports.bufferPoolClear();
  }

  /** Get buffer pool statistics. */
  cacheStats(): { count: number; bytes: number; maxBytes: number } {
    const outPtr = this.exports.alloc(12);
    if (!outPtr) return { count: 0, bytes: 0, maxBytes: 0 };
    this.exports.bufferPoolGetStats(outPtr, outPtr + 4, outPtr + 8);
    const view = new DataView(this.exports.memory.buffer);
    return {
      count: view.getUint32(outPtr, true),
      bytes: view.getUint32(outPtr + 4, true),
      maxBytes: view.getUint32(outPtr + 8, true),
    };
  }

  executeQuery(query: QueryDescriptor): Row[] | null {
    const MAX_SQL_LENGTH = 64 * 1024; // 64KB — WASM SQL input buffer is fixed-size
    const sqlBytes = textEncoder.encode(queryToSql(query));
    if (sqlBytes.length > MAX_SQL_LENGTH) {
      throw new Error(`SQL query too large (${sqlBytes.length} bytes, max ${MAX_SQL_LENGTH})`);
    }
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

  /**
   * Execute a query and return the result as QMCB columnar binary.
   * Bypasses Row[] creation entirely — reads WASM memory and writes QMCB
   * in a single pass (memcpy for numeric columns).
   */
  executeQueryColumnar(query: QueryDescriptor): ArrayBuffer | null {
    const MAX_SQL_LENGTH = 64 * 1024;
    const sqlBytes = textEncoder.encode(queryToSql(query));
    if (sqlBytes.length > MAX_SQL_LENGTH) {
      throw new Error(`SQL query too large (${sqlBytes.length} bytes, max ${MAX_SQL_LENGTH})`);
    }
    const sqlBufPtr = this.exports.getSqlInputBuffer();
    new Uint8Array(this.exports.memory.buffer, sqlBufPtr, sqlBytes.length).set(sqlBytes);
    this.exports.setSqlInputLength(sqlBytes.length);

    const resultPtr = this.exports.executeSql();
    if (!resultPtr) return null;
    const resultSize = this.exports.getResultSize();
    if (!resultSize) return null;

    // Convert directly from WASM memory to QMCB — no Row[] intermediate
    const qmcb = wasmResultToQMCB(this.exports.memory.buffer, resultPtr, resultSize);
    this.exports.resetResult();
    return qmcb;
  }

  /** SIMD sum of Float64 buffer. Returns 0 for empty input. */
  sumFloat64(buf: ArrayBuffer): number {
    if (buf.byteLength === 0) return 0;
    const numElements = buf.byteLength >> 3;
    const ptr = this.exports.alloc(buf.byteLength);
    if (!ptr) return 0;
    new Uint8Array(this.exports.memory.buffer, ptr, buf.byteLength).set(new Uint8Array(buf));
    return this.exports.sumFloat64Buffer(ptr >> 3, numElements);
  }

  /** SIMD min of Float64 buffer. Returns Infinity for empty input. */
  minFloat64(buf: ArrayBuffer): number {
    if (buf.byteLength === 0) return Infinity;
    const numElements = buf.byteLength >> 3;
    const ptr = this.exports.alloc(buf.byteLength);
    if (!ptr) return Infinity;
    new Uint8Array(this.exports.memory.buffer, ptr, buf.byteLength).set(new Uint8Array(buf));
    return this.exports.minFloat64Buffer(ptr >> 3, numElements);
  }

  /** SIMD max of Float64 buffer. Returns -Infinity for empty input. */
  maxFloat64(buf: ArrayBuffer): number {
    if (buf.byteLength === 0) return -Infinity;
    const numElements = buf.byteLength >> 3;
    const ptr = this.exports.alloc(buf.byteLength);
    if (!ptr) return -Infinity;
    new Uint8Array(this.exports.memory.buffer, ptr, buf.byteLength).set(new Uint8Array(buf));
    return this.exports.maxFloat64Buffer(ptr >> 3, numElements);
  }

  /** SIMD avg of Float64 buffer. Returns 0 for empty input. */
  avgFloat64(buf: ArrayBuffer): number {
    if (buf.byteLength === 0) return 0;
    const numElements = buf.byteLength >> 3;
    const ptr = this.exports.alloc(buf.byteLength);
    if (!ptr) return 0;
    new Uint8Array(this.exports.memory.buffer, ptr, buf.byteLength).set(new Uint8Array(buf));
    return this.exports.avgFloat64Buffer(ptr >> 3, numElements);
  }

  /** SIMD sum of Int64 buffer. Returns 0n for empty input. */
  sumInt64(buf: ArrayBuffer): bigint {
    if (buf.byteLength === 0) return 0n;
    const numElements = buf.byteLength >> 3;
    const ptr = this.exports.alloc(buf.byteLength);
    if (!ptr) return 0n;
    new Uint8Array(this.exports.memory.buffer, ptr, buf.byteLength).set(new Uint8Array(buf));
    return this.exports.sumInt64Buffer(ptr >> 3, numElements);
  }

  /** SIMD min of Int64 buffer. Returns MAX_SAFE_INTEGER for empty input. */
  minInt64(buf: ArrayBuffer): bigint {
    if (buf.byteLength === 0) return BigInt(Number.MAX_SAFE_INTEGER);
    const numElements = buf.byteLength >> 3;
    const ptr = this.exports.alloc(buf.byteLength);
    if (!ptr) return BigInt(Number.MAX_SAFE_INTEGER);
    new Uint8Array(this.exports.memory.buffer, ptr, buf.byteLength).set(new Uint8Array(buf));
    return this.exports.minInt64Buffer(ptr >> 3, numElements);
  }

  /** SIMD max of Int64 buffer. Returns MIN_SAFE_INTEGER for empty input. */
  maxInt64(buf: ArrayBuffer): bigint {
    if (buf.byteLength === 0) return BigInt(Number.MIN_SAFE_INTEGER);
    const numElements = buf.byteLength >> 3;
    const ptr = this.exports.alloc(buf.byteLength);
    if (!ptr) return BigInt(Number.MIN_SAFE_INTEGER);
    new Uint8Array(this.exports.memory.buffer, ptr, buf.byteLength).set(new Uint8Array(buf));
    return this.exports.maxInt64Buffer(ptr >> 3, numElements);
  }

  /** Load a fragment file into the WASM fragment reader. */
  loadFragment(data: ArrayBuffer): boolean {
    const ptr = this.exports.alloc(data.byteLength);
    if (!ptr) return false;
    new Uint8Array(this.exports.memory.buffer, ptr, data.byteLength).set(new Uint8Array(data));
    return this.exports.fragmentLoad(ptr, data.byteLength) === 0;
  }

  /** Read a typed column from a loaded fragment. */
  readFragmentColumn(colIdx: number, dtype: string): Float64Array | Float32Array | Int32Array | BigInt64Array | null {
    const rowCount = Number(this.exports.fragmentGetRowCount());
    if (rowCount === 0) return null;

    switch (dtype) {
      case "float64": {
        const outPtr = this.exports.alloc(rowCount * 8);
        if (!outPtr) return null;
        const read = this.exports.fragmentReadFloat64(colIdx, outPtr, rowCount);
        return new Float64Array(this.exports.memory.buffer.slice(outPtr, outPtr + read * 8));
      }
      case "float32": {
        const outPtr = this.exports.alloc(rowCount * 4);
        if (!outPtr) return null;
        const read = this.exports.fragmentReadFloat32(colIdx, outPtr, rowCount);
        return new Float32Array(this.exports.memory.buffer.slice(outPtr, outPtr + read * 4));
      }
      case "int32": {
        const outPtr = this.exports.alloc(rowCount * 4);
        if (!outPtr) return null;
        const read = this.exports.fragmentReadInt32(colIdx, outPtr, rowCount);
        return new Int32Array(this.exports.memory.buffer.slice(outPtr, outPtr + read * 4));
      }
      case "int64": {
        const outPtr = this.exports.alloc(rowCount * 8);
        if (!outPtr) return null;
        const read = this.exports.fragmentReadInt64(colIdx, outPtr, rowCount);
        return new BigInt64Array(this.exports.memory.buffer.slice(outPtr, outPtr + read * 8));
      }
      default: return null;
    }
  }

  /**
   * Build a Lance fragment file from column data.
   * Returns the raw Lance binary bytes ready to write to R2/disk.
   */
  buildFragment(columns: { name: string; dtype: string; values: ArrayBufferLike }[]): Uint8Array {
    // Estimate capacity: sum of all values plus overhead
    let totalBytes = 0;
    for (const col of columns) totalBytes += col.values.byteLength;
    const capacity = totalBytes + columns.length * 256 + 4096; // metadata overhead

    if (!this.exports.fragmentBegin(capacity)) {
      throw new Error("WASM fragmentBegin failed — OOM or init error");
    }

    for (const col of columns) {
      const { ptr: namePtr, len: nameLen } = this.writeString(col.name);
      if (!namePtr) throw new Error(`WASM OOM writing column name "${col.name}"`);

      let result = 0;
      switch (col.dtype) {
        case "int64": {
          const i64 = new BigInt64Array(col.values);
          const dataPtr = this.exports.alloc(col.values.byteLength);
          if (!dataPtr) throw new Error("WASM OOM");
          new Uint8Array(this.exports.memory.buffer, dataPtr, col.values.byteLength)
            .set(new Uint8Array(col.values instanceof ArrayBuffer ? col.values : col.values.slice(0)));
          result = this.exports.fragmentAddInt64Column(namePtr, nameLen, dataPtr, i64.length, 0);
          break;
        }
        case "int32": {
          const i32 = new Int32Array(col.values);
          const dataPtr = this.exports.alloc(col.values.byteLength);
          if (!dataPtr) throw new Error("WASM OOM");
          new Uint8Array(this.exports.memory.buffer, dataPtr, col.values.byteLength)
            .set(new Uint8Array(col.values instanceof ArrayBuffer ? col.values : col.values.slice(0)));
          result = this.exports.fragmentAddInt32Column(namePtr, nameLen, dataPtr, i32.length, 0);
          break;
        }
        case "float64": {
          const f64 = new Float64Array(col.values);
          const dataPtr = this.exports.alloc(col.values.byteLength);
          if (!dataPtr) throw new Error("WASM OOM");
          new Uint8Array(this.exports.memory.buffer, dataPtr, col.values.byteLength)
            .set(new Uint8Array(col.values instanceof ArrayBuffer ? col.values : col.values.slice(0)));
          result = this.exports.fragmentAddFloat64Column(namePtr, nameLen, dataPtr, f64.length, 0);
          break;
        }
        case "float32": {
          const f32 = new Float32Array(col.values);
          const dataPtr = this.exports.alloc(col.values.byteLength);
          if (!dataPtr) throw new Error("WASM OOM");
          new Uint8Array(this.exports.memory.buffer, dataPtr, col.values.byteLength)
            .set(new Uint8Array(col.values instanceof ArrayBuffer ? col.values : col.values.slice(0)));
          result = this.exports.fragmentAddFloat32Column(namePtr, nameLen, dataPtr, f32.length, 0);
          break;
        }
        case "utf8": case "string": {
          // Values are length-prefixed strings. Scan to extract raw string data + build offsets array.
          const srcBuf = col.values instanceof ArrayBuffer ? col.values : col.values.slice(0);
          const view = new DataView(srcBuf);
          // First pass: count strings and compute total raw string bytes
          let count = 0, pos = 0, totalStrBytes = 0;
          while (pos + 4 <= srcBuf.byteLength) {
            const strLen = view.getUint32(pos, true);
            pos += 4 + strLen;
            totalStrBytes += strLen;
            count++;
          }
          // Build raw string data (without length prefixes) and offsets array
          const rawStrData = new Uint8Array(totalStrBytes);
          const offsets = new Uint32Array(count + 1); // Arrow-style: count+1 offsets
          pos = 0;
          let strOff = 0;
          for (let si = 0; si < count; si++) {
            const strLen = view.getUint32(pos, true);
            pos += 4;
            offsets[si] = strOff;
            rawStrData.set(new Uint8Array(srcBuf, pos, strLen), strOff);
            strOff += strLen;
            pos += strLen;
          }
          offsets[count] = strOff;
          // Copy raw string data to WASM
          const dataPtr = this.exports.alloc(totalStrBytes);
          if (!dataPtr) throw new Error("WASM OOM");
          new Uint8Array(this.exports.memory.buffer, dataPtr, totalStrBytes).set(rawStrData);
          // Copy offsets to WASM
          const offsetsPtr = this.exports.alloc(offsets.byteLength);
          if (!offsetsPtr) throw new Error("WASM OOM");
          new Uint8Array(this.exports.memory.buffer, offsetsPtr, offsets.byteLength)
            .set(new Uint8Array(offsets.buffer));
          result = this.exports.fragmentAddStringColumn(
            namePtr, nameLen, dataPtr, totalStrBytes, offsetsPtr / 4, count, 0,
          );
          break;
        }
        case "bool": {
          const dataPtr = this.exports.alloc(col.values.byteLength);
          if (!dataPtr) throw new Error("WASM OOM");
          new Uint8Array(this.exports.memory.buffer, dataPtr, col.values.byteLength)
            .set(new Uint8Array(col.values instanceof ArrayBuffer ? col.values : col.values.slice(0)));
          const byteCount = col.values.byteLength;
          // rowCount must be provided externally for exact count; default to byteLength * 8
          const rowCount = (col as { rowCount?: number }).rowCount ?? byteCount * 8;
          result = this.exports.fragmentAddBoolColumn(namePtr, nameLen, dataPtr, byteCount, rowCount, 0);
          break;
        }
        default:
          throw new Error(`Unsupported dtype for fragment building: ${col.dtype}`);
      }

      if (!result) throw new Error(`WASM failed to add column "${col.name}" (dtype: ${col.dtype})`);
    }

    const bytesWritten = this.exports.fragmentEnd();
    if (!bytesWritten) throw new Error("WASM fragmentEnd failed");

    const bufPtr = this.exports.writerGetBuffer();
    if (!bufPtr) throw new Error("WASM writerGetBuffer returned null");

    return new Uint8Array(this.exports.memory.buffer, bufPtr, bytesWritten).slice();
  }

  // --- IVF-PQ vector index ---

  /** Load a serialized IVF-PQ index into WASM. Returns a handle (>0) or throws on failure. */
  loadIvfPqIndex(indexData: ArrayBuffer): number {
    const ptr = this.exports.alloc(indexData.byteLength);
    if (!ptr) throw new Error("WASM OOM allocating IVF-PQ index data");
    new Uint8Array(this.exports.memory.buffer, ptr, indexData.byteLength).set(new Uint8Array(indexData));
    const handle = this.exports.ivfPqLoadIndex(ptr, indexData.byteLength);
    if (!handle) throw new Error("Failed to load IVF-PQ index — invalid format or no free slots");
    return handle;
  }

  /** Search an IVF-PQ index for nearest neighbors. */
  searchIvfPq(
    handle: number, query: Float32Array, topK: number, nprobe = 0,
  ): { indices: Uint32Array; scores: Float32Array } {
    const dim = query.length;
    const totalBytes = query.byteLength + topK * 4 + topK * 4;
    const basePtr = this.exports.alloc(totalBytes);
    if (!basePtr) return { indices: new Uint32Array(0), scores: new Float32Array(0) };

    const qPtr = basePtr;
    new Uint8Array(this.exports.memory.buffer, qPtr, query.byteLength).set(
      new Uint8Array(query.buffer, query.byteOffset, query.byteLength),
    );
    const iPtr = qPtr + query.byteLength;
    const sPtr = iPtr + topK * 4;

    const count = this.exports.ivfPqSearch(
      handle, qPtr / 4, dim, topK, nprobe, iPtr / 4, sPtr / 4,
    );

    return {
      indices: new Uint32Array(this.exports.memory.buffer.slice(iPtr, iPtr + count * 4)),
      scores: new Float32Array(this.exports.memory.buffer.slice(sPtr, sPtr + count * 4)),
    };
  }

  /** Free an IVF-PQ index handle. */
  freeIvfPqIndex(handle: number): void {
    this.exports.ivfPqFree(handle);
  }

  vectorSearchBuffer(
    vectors: Float32Array, numVectors: number, dim: number,
    queryVector: Float32Array, topK: number, normalized = false,
  ): { indices: Uint32Array; scores: Float32Array } {
    const totalBytes = vectors.byteLength + queryVector.byteLength + topK * 8;
    const basePtr = this.exports.alloc(totalBytes);
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

const COMPARISON_OP_MAP: Record<string, string> = { eq: "=", neq: "!=", gt: ">", gte: ">=", lt: "<", lte: "<=" };

function filterToSql(f: FilterOp): string {
  if (f.op === "is_null") return `${quote(f.column)} IS NULL`;
  if (f.op === "is_not_null") return `${quote(f.column)} IS NOT NULL`;
  if (f.op === "in" && Array.isArray(f.value)) {
    return `${quote(f.column)} IN (${f.value.map(v => typeof v === "string" ? `'${escapeSql(v)}'` : String(v)).join(", ")})`;
  }
  if (f.op === "not_in" && Array.isArray(f.value)) {
    return `${quote(f.column)} NOT IN (${f.value.map(v => typeof v === "string" ? `'${escapeSql(v)}'` : String(v)).join(", ")})`;
  }
  if ((f.op === "between" || f.op === "not_between") && Array.isArray(f.value) && f.value.length === 2) {
    const lo = typeof f.value[0] === "string" ? `'${escapeSql(f.value[0])}'` : String(f.value[0]);
    const hi = typeof f.value[1] === "string" ? `'${escapeSql(f.value[1])}'` : String(f.value[1]);
    return `${quote(f.column)} ${f.op === "not_between" ? "NOT " : ""}BETWEEN ${lo} AND ${hi}`;
  }
  if (f.op === "like") return `${quote(f.column)} LIKE '${escapeSql(f.value as string)}'`;
  if (f.op === "not_like") return `${quote(f.column)} NOT LIKE '${escapeSql(f.value as string)}'`;
  const val = typeof f.value === "string" ? `'${escapeSql(f.value)}'` : String(f.value);
  return `${quote(f.column)} ${COMPARISON_OP_MAP[f.op] ?? "="} ${val}`;
}

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
    parts.push(`SELECT ${query.projections.map(quote).join(", ")}`);
  } else {
    parts.push("SELECT *");
  }

  parts.push(`FROM ${quote(query.table)}`);

  const andConditions = query.filters.map(filterToSql);
  const orGroupConditions = (query.filterGroups ?? [])
    .filter(g => g.length > 0)
    .map(g => g.length === 1 ? filterToSql(g[0]) : `(${g.map(filterToSql).join(" AND ")})`);
  if (andConditions.length > 0 || orGroupConditions.length > 0) {
    const whereParts: string[] = [...andConditions];
    if (orGroupConditions.length > 0) {
      whereParts.push(orGroupConditions.length === 1 ? orGroupConditions[0] : `(${orGroupConditions.join(" OR ")})`);
    }
    parts.push(`WHERE ${whereParts.join(" AND ")}`);
  }

  if (query.vectorSearch) {
    const vs = query.vectorSearch;
    const vecStr = `[${Array.from(vs.queryVector).join(",")}]`;
    parts.push(`${query.filters.length > 0 ? "AND" : "WHERE"} ${quote(vs.column)} NEAR ${vecStr}`);
  }

  if (query.groupBy?.length) parts.push(`GROUP BY ${query.groupBy.map(quote).join(", ")}`);
  if (query.sortColumn) parts.push(`ORDER BY ${quote(query.sortColumn)} ${query.sortDirection?.toUpperCase() ?? "ASC"}`);
  if (query.limit) parts.push(`LIMIT ${query.limit}`);
  if (query.offset) parts.push(`OFFSET ${query.offset}`);

  return parts.join(" ");
}

function quote(name: string): string {
  return /^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name) ? name : `"${name.replace(/"/g, '""')}"`;
}

function escapeSql(value: string): string {
  return value.replace(/'/g, "''");
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

  const rows: Row[] = new Array(numRows);
  for (let i = 0; i < numRows; i++) rows[i] = {};
  let dp = dataStart;
  const end = size;

  for (const col of columns) {
    for (let r = 0; r < numRows; r++) {
      if (dp + 1 > end) return rows.slice(0, r > 0 ? r : 0); // truncated result
      switch (col.type) {
        case WasmColumnType.Int64:   if (dp + 8 > end) return rows; rows[r][col.name] = view.getBigInt64(dp, true); dp += 8; break;
        case WasmColumnType.Float64: if (dp + 8 > end) return rows; rows[r][col.name] = view.getFloat64(dp, true); dp += 8; break;
        case WasmColumnType.Int32:   if (dp + 4 > end) return rows; rows[r][col.name] = view.getInt32(dp, true); dp += 4; break;
        case WasmColumnType.Float32: if (dp + 4 > end) return rows; rows[r][col.name] = view.getFloat32(dp, true); dp += 4; break;
        case WasmColumnType.Bool:    rows[r][col.name] = buf[dp] !== 0; dp += 1; break;
        case WasmColumnType.String: {
          if (dp + 4 > end) return rows;
          const len = view.getUint32(dp, true); dp += 4;
          if (dp + len > end) return rows;
          rows[r][col.name] = textDecoder.decode(buf.subarray(dp, dp + len)); dp += len;
          break;
        }
        default: rows[r][col.name] = null; dp += 8; break;
      }
    }
  }

  return rows;
}
