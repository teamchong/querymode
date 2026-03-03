/**
 * EdgeQ WASM engine — wraps Zig SIMD compute layer.
 * TS handles orchestration (R2 reads, cache); WASM handles all compute.
 */

import type { Row, DataType, PageInfo } from "./types.js";
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
  intersectIndices(aPtr: number, aLen: number, bPtr: number, bLen: number, outPtr: number, maxOut: number): number;

  // Buffer SIMD aggregates (aggregates.zig) — ptr in element units, len = element count
  sumFloat64Buffer(ptr: number, len: number): number;
  minFloat64Buffer(ptr: number, len: number): number;
  maxFloat64Buffer(ptr: number, len: number): number;
  avgFloat64Buffer(ptr: number, len: number): number;
  sumInt32Buffer(ptr: number, len: number): number;
  sumInt64Buffer(ptr: number, len: number): bigint;
  minInt64Buffer(ptr: number, len: number): bigint;
  maxInt64Buffer(ptr: number, len: number): bigint;

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

  /** Decompress ZSTD data using the Zig std.compress.zstd implementation. */
  decompressZstd(compressed: Uint8Array): Uint8Array {
    const inPtr = this.exports.wasmAlloc(compressed.length);
    if (!inPtr) throw new Error("WASM OOM allocating zstd input");
    new Uint8Array(this.exports.memory.buffer, inPtr, compressed.length).set(compressed);

    const decompressedSize = this.exports.zstd_get_decompressed_size(inPtr, compressed.length);
    const capacity = decompressedSize || compressed.length * 4; // estimate if unknown
    const outPtr = this.exports.wasmAlloc(capacity);
    if (!outPtr) throw new Error("WASM OOM allocating zstd output");

    const written = this.exports.zstd_decompress(inPtr, compressed.length, outPtr, capacity);
    if (written === 0 && compressed.length > 0) throw new Error("zstd decompression failed");
    return new Uint8Array(this.exports.memory.buffer, outPtr, written).slice();
  }

  /** Decompress GZIP data using the Zig std.compress.gzip implementation. */
  decompressGzip(compressed: Uint8Array): Uint8Array {
    const inPtr = this.exports.wasmAlloc(compressed.length);
    if (!inPtr) throw new Error("WASM OOM allocating gzip input");
    new Uint8Array(this.exports.memory.buffer, inPtr, compressed.length).set(compressed);

    const capacity = compressed.length * 4;
    const outPtr = this.exports.wasmAlloc(capacity);
    if (!outPtr) throw new Error("WASM OOM allocating gzip output");

    const written = this.exports.gzip_decompress(inPtr, compressed.length, outPtr, capacity);
    if (written === 0 && compressed.length > 0) throw new Error("gzip decompression failed");
    return new Uint8Array(this.exports.memory.buffer, outPtr, written).slice();
  }

  /** Decompress LZ4 block data (Parquet hadoop codec). */
  decompressLz4(compressed: Uint8Array, uncompressedSize: number): Uint8Array {
    const inPtr = this.exports.wasmAlloc(compressed.length);
    if (!inPtr) throw new Error("WASM OOM allocating lz4 input");
    new Uint8Array(this.exports.memory.buffer, inPtr, compressed.length).set(compressed);

    const outPtr = this.exports.wasmAlloc(uncompressedSize);
    if (!outPtr) throw new Error("WASM OOM allocating lz4 output");

    const written = this.exports.lz4_block_decompress(inPtr, compressed.length, outPtr, uncompressedSize);
    if (written === 0 && compressed.length > 0) throw new Error("lz4 decompression failed");
    return new Uint8Array(this.exports.memory.buffer, outPtr, written).slice();
  }

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
        const dataPtr = this.exports.wasmAlloc(flat.byteLength);
        if (!dataPtr) return false;
        new Uint8Array(this.exports.memory.buffer, dataPtr, flat.byteLength).set(flat);
        this.exports.registerTableInt64(tPtr, tLen, cPtr, cLen, dataPtr, totalRows);
        return true;
      }

      case "float64": {
        const dataPtr = this.exports.wasmAlloc(flat.byteLength);
        if (!dataPtr) return false;
        new Uint8Array(this.exports.memory.buffer, dataPtr, flat.byteLength).set(flat);
        this.exports.registerTableFloat64(tPtr, tLen, cPtr, cLen, dataPtr, totalRows);
        return true;
      }

      case "float32": {
        if (dtype === "float32" && listDim && listDim > 0) {
          // fixed_size_list path handled below
          break;
        }
        // Promote float32 → float64
        const src = new Float32Array(flat.buffer, flat.byteOffset, flat.byteLength >> 2);
        const f64Bytes = totalRows * 8;
        const dataPtr = this.exports.wasmAlloc(f64Bytes);
        if (!dataPtr) return false;
        const dst = new Float64Array(this.exports.memory.buffer, dataPtr, totalRows);
        for (let i = 0; i < totalRows; i++) dst[i] = src[i];
        this.exports.registerTableFloat64(tPtr, tLen, cPtr, cLen, dataPtr, totalRows);
        return true;
      }

      case "int32": case "uint32": {
        // Promote to i64
        const src = dtype === "int32"
          ? new Int32Array(flat.buffer, flat.byteOffset, flat.byteLength >> 2)
          : new Uint32Array(flat.buffer, flat.byteOffset, flat.byteLength >> 2);
        const i64Bytes = totalRows * 8;
        const dataPtr = this.exports.wasmAlloc(i64Bytes);
        if (!dataPtr) return false;
        const dst = new BigInt64Array(this.exports.memory.buffer, dataPtr, totalRows);
        for (let i = 0; i < totalRows; i++) dst[i] = BigInt(src[i]);
        this.exports.registerTableInt64(tPtr, tLen, cPtr, cLen, dataPtr, totalRows);
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
        const i64Bytes = totalRows * 8;
        const dataPtr = this.exports.wasmAlloc(i64Bytes);
        if (!dataPtr) return false;
        const dst = new BigInt64Array(this.exports.memory.buffer, dataPtr, totalRows);
        for (let i = 0; i < totalRows; i++) dst[i] = BigInt(src[i]);
        this.exports.registerTableInt64(tPtr, tLen, cPtr, cLen, dataPtr, totalRows);
        return true;
      }

      case "uint64": {
        // Cast to signed i64
        const dataPtr = this.exports.wasmAlloc(flat.byteLength);
        if (!dataPtr) return false;
        new Uint8Array(this.exports.memory.buffer, dataPtr, flat.byteLength).set(flat);
        this.exports.registerTableInt64(tPtr, tLen, cPtr, cLen, dataPtr, totalRows);
        return true;
      }

      case "bool": {
        // Unpack bitmap to i64 (0/1)
        const i64Bytes = totalRows * 8;
        const dataPtr = this.exports.wasmAlloc(i64Bytes);
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
        const dataPtr = this.exports.wasmAlloc(flat.byteLength);
        if (!dataPtr) return false;
        new Uint8Array(this.exports.memory.buffer, dataPtr, flat.byteLength).set(flat);
        this.exports.registerTableFloat32Vector(tPtr, tLen, cPtr, cLen, dataPtr, totalRows, dim);
        return true;
      }

      case "utf8": {
        // Scan length-prefixed strings to build offset/length arrays
        const offsets = new Uint32Array(totalRows);
        const lengths = new Uint32Array(totalRows);
        let dataLen = 0;
        let pos = 0;
        const view = new DataView(flat.buffer, flat.byteOffset, flat.byteLength);
        let ri = 0;
        while (pos + 4 <= flat.byteLength && ri < totalRows) {
          const strLen = view.getUint32(pos, true);
          pos += 4;
          offsets[ri] = dataLen;
          lengths[ri] = strLen;
          dataLen += strLen;
          pos += strLen;
          ri++;
        }

        // Copy string data (without length prefixes) to WASM
        const strData = new Uint8Array(dataLen);
        pos = 0;
        let dOff = 0;
        const view2 = new DataView(flat.buffer, flat.byteOffset, flat.byteLength);
        for (let i = 0; i < ri; i++) {
          const strLen = view2.getUint32(pos, true);
          pos += 4;
          strData.set(flat.subarray(pos, pos + strLen), dOff);
          dOff += strLen;
          pos += strLen;
        }

        const offsetsPtr = this.exports.wasmAlloc(offsets.byteLength);
        if (!offsetsPtr) return false;
        new Uint32Array(this.exports.memory.buffer, offsetsPtr, ri).set(offsets.subarray(0, ri));

        const lengthsPtr = this.exports.wasmAlloc(lengths.byteLength);
        if (!lengthsPtr) return false;
        new Uint32Array(this.exports.memory.buffer, lengthsPtr, ri).set(lengths.subarray(0, ri));

        const dataPtr = this.exports.wasmAlloc(dataLen || 1);
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

    // fixed_size_list when reached via float32 break
    if (dtype === "float32" && listDim && listDim > 0) {
      const dim = listDim;
      const dataPtr = this.exports.wasmAlloc(flat.byteLength);
      if (!dataPtr) return false;
      new Uint8Array(this.exports.memory.buffer, dataPtr, flat.byteLength).set(flat);
      this.exports.registerTableFloat32Vector(tPtr, tLen, cPtr, cLen, dataPtr, totalRows, dim);
      return true;
    }

    return true;
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
    const dataPtr = this.exports.wasmAlloc(data.byteLength);
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
    const outPtr = this.exports.wasmAlloc(8);
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
    const outPtr = this.exports.wasmAlloc(12);
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

  /** SIMD sum of Float64 buffer. Returns 0 for empty input. */
  sumFloat64(buf: ArrayBuffer): number {
    if (buf.byteLength === 0) return 0;
    const numElements = buf.byteLength >> 3;
    const ptr = this.exports.wasmAlloc(buf.byteLength);
    if (!ptr) return 0;
    new Uint8Array(this.exports.memory.buffer, ptr, buf.byteLength).set(new Uint8Array(buf));
    return this.exports.sumFloat64Buffer(ptr >> 3, numElements);
  }

  /** SIMD min of Float64 buffer. Returns Infinity for empty input. */
  minFloat64(buf: ArrayBuffer): number {
    if (buf.byteLength === 0) return Infinity;
    const numElements = buf.byteLength >> 3;
    const ptr = this.exports.wasmAlloc(buf.byteLength);
    if (!ptr) return Infinity;
    new Uint8Array(this.exports.memory.buffer, ptr, buf.byteLength).set(new Uint8Array(buf));
    return this.exports.minFloat64Buffer(ptr >> 3, numElements);
  }

  /** SIMD max of Float64 buffer. Returns -Infinity for empty input. */
  maxFloat64(buf: ArrayBuffer): number {
    if (buf.byteLength === 0) return -Infinity;
    const numElements = buf.byteLength >> 3;
    const ptr = this.exports.wasmAlloc(buf.byteLength);
    if (!ptr) return -Infinity;
    new Uint8Array(this.exports.memory.buffer, ptr, buf.byteLength).set(new Uint8Array(buf));
    return this.exports.maxFloat64Buffer(ptr >> 3, numElements);
  }

  /** SIMD avg of Float64 buffer. Returns 0 for empty input. */
  avgFloat64(buf: ArrayBuffer): number {
    if (buf.byteLength === 0) return 0;
    const numElements = buf.byteLength >> 3;
    const ptr = this.exports.wasmAlloc(buf.byteLength);
    if (!ptr) return 0;
    new Uint8Array(this.exports.memory.buffer, ptr, buf.byteLength).set(new Uint8Array(buf));
    return this.exports.avgFloat64Buffer(ptr >> 3, numElements);
  }

  /** Load a fragment file into the WASM fragment reader. */
  loadFragment(data: ArrayBuffer): boolean {
    const ptr = this.exports.wasmAlloc(data.byteLength);
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
        const outPtr = this.exports.wasmAlloc(rowCount * 8);
        if (!outPtr) return null;
        const read = this.exports.fragmentReadFloat64(colIdx, outPtr, rowCount);
        return new Float64Array(this.exports.memory.buffer.slice(outPtr, outPtr + read * 8));
      }
      case "float32": {
        const outPtr = this.exports.wasmAlloc(rowCount * 4);
        if (!outPtr) return null;
        const read = this.exports.fragmentReadFloat32(colIdx, outPtr, rowCount);
        return new Float32Array(this.exports.memory.buffer.slice(outPtr, outPtr + read * 4));
      }
      case "int32": {
        const outPtr = this.exports.wasmAlloc(rowCount * 4);
        if (!outPtr) return null;
        const read = this.exports.fragmentReadInt32(colIdx, outPtr, rowCount);
        return new Int32Array(this.exports.memory.buffer.slice(outPtr, outPtr + read * 4));
      }
      case "int64": {
        const outPtr = this.exports.wasmAlloc(rowCount * 8);
        if (!outPtr) return null;
        const read = this.exports.fragmentReadInt64(colIdx, outPtr, rowCount);
        return new BigInt64Array(this.exports.memory.buffer.slice(outPtr, outPtr + read * 8));
      }
      default: return null;
    }
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
