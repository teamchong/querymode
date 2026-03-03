/**
 * Bridge between TS page-streaming and WASM SIMD compute.
 * WASM is required — no fallback paths.
 */

import type { DataType } from "./types.js";
import type { WasmEngine } from "./wasm-engine.js";
import { decodePage } from "./decode.js";

export interface PageProcessor {
  sumFloat64(buf: ArrayBuffer): number;
  minFloat64(buf: ArrayBuffer): number;
  maxFloat64(buf: ArrayBuffer): number;
  vectorSearchPage(
    vectors: Float32Array,
    numVecs: number,
    dim: number,
    query: Float32Array,
    topK: number,
  ): { indices: Uint32Array; scores: Float32Array };
  decodePage(
    rawBuf: ArrayBuffer,
    dtype: DataType,
    nullCount?: number,
    rowCount?: number,
  ): (number | bigint | string | null)[];
}

export function createPageProcessor(wasm: WasmEngine): PageProcessor {
  return {
    sumFloat64(buf: ArrayBuffer): number {
      const f64 = new Float64Array(buf);
      if (f64.length === 0) return 0;
      const ptr = wasm.exports.wasmAlloc(f64.byteLength);
      if (!ptr) throw new Error("WASM alloc failed for sumFloat64");
      new Uint8Array(wasm.exports.memory.buffer, ptr, f64.byteLength).set(
        new Uint8Array(buf, 0, f64.byteLength),
      );
      return wasm.exports.sumFloat64Buffer(ptr >> 3, f64.length);
    },

    minFloat64(buf: ArrayBuffer): number {
      const f64 = new Float64Array(buf);
      if (f64.length === 0) return Infinity;
      const ptr = wasm.exports.wasmAlloc(f64.byteLength);
      if (!ptr) throw new Error("WASM alloc failed for minFloat64");
      new Uint8Array(wasm.exports.memory.buffer, ptr, f64.byteLength).set(
        new Uint8Array(buf, 0, f64.byteLength),
      );
      return wasm.exports.minFloat64Buffer(ptr >> 3, f64.length);
    },

    maxFloat64(buf: ArrayBuffer): number {
      const f64 = new Float64Array(buf);
      if (f64.length === 0) return -Infinity;
      const ptr = wasm.exports.wasmAlloc(f64.byteLength);
      if (!ptr) throw new Error("WASM alloc failed for maxFloat64");
      new Uint8Array(wasm.exports.memory.buffer, ptr, f64.byteLength).set(
        new Uint8Array(buf, 0, f64.byteLength),
      );
      return wasm.exports.maxFloat64Buffer(ptr >> 3, f64.length);
    },

    vectorSearchPage(
      vectors: Float32Array,
      numVecs: number,
      dim: number,
      query: Float32Array,
      topK: number,
    ): { indices: Uint32Array; scores: Float32Array } {
      return wasm.vectorSearchBuffer(vectors, numVecs, dim, query, topK);
    },

    decodePage(
      rawBuf: ArrayBuffer,
      dtype: DataType,
      nullCount?: number,
      rowCount?: number,
    ): (number | bigint | string | null)[] {
      return decodePage(rawBuf, dtype, nullCount, rowCount);
    },
  };
}
