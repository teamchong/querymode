/**
 * Bridge between TS page-streaming and WASM SIMD compute.
 * When WASM is available, delegates numeric ops to SIMD kernels.
 * Falls back to pure TS for environments without WASM.
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

export function createPageProcessor(wasm: WasmEngine | null): PageProcessor {
  if (wasm) {
    return createWasmProcessor(wasm);
  }
  return createTsFallbackProcessor();
}

// --- WASM-backed processor ---

function createWasmProcessor(wasm: WasmEngine): PageProcessor {
  return {
    sumFloat64(buf: ArrayBuffer): number {
      const f64 = new Float64Array(buf);
      if (f64.length === 0) return 0;
      const ptr = wasm.exports.wasmAlloc(f64.byteLength);
      if (!ptr) return tsFallbackSum(f64);
      new Uint8Array(wasm.exports.memory.buffer, ptr, f64.byteLength).set(
        new Uint8Array(buf, 0, f64.byteLength),
      );
      return wasm.exports.sumFloat64Buffer(ptr >> 3, f64.length);
    },

    minFloat64(buf: ArrayBuffer): number {
      const f64 = new Float64Array(buf);
      if (f64.length === 0) return Infinity;
      const ptr = wasm.exports.wasmAlloc(f64.byteLength);
      if (!ptr) return tsFallbackMin(f64);
      new Uint8Array(wasm.exports.memory.buffer, ptr, f64.byteLength).set(
        new Uint8Array(buf, 0, f64.byteLength),
      );
      return wasm.exports.minFloat64Buffer(ptr >> 3, f64.length);
    },

    maxFloat64(buf: ArrayBuffer): number {
      const f64 = new Float64Array(buf);
      if (f64.length === 0) return -Infinity;
      const ptr = wasm.exports.wasmAlloc(f64.byteLength);
      if (!ptr) return tsFallbackMax(f64);
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

// --- Pure TS fallback processor ---

function createTsFallbackProcessor(): PageProcessor {
  return {
    sumFloat64(buf: ArrayBuffer): number {
      return tsFallbackSum(new Float64Array(buf));
    },

    minFloat64(buf: ArrayBuffer): number {
      const f64 = new Float64Array(buf);
      if (f64.length === 0) return Infinity;
      return tsFallbackMin(f64);
    },

    maxFloat64(buf: ArrayBuffer): number {
      const f64 = new Float64Array(buf);
      if (f64.length === 0) return -Infinity;
      return tsFallbackMax(f64);
    },

    vectorSearchPage(
      vectors: Float32Array,
      numVecs: number,
      dim: number,
      query: Float32Array,
      topK: number,
    ): { indices: Uint32Array; scores: Float32Array } {
      return tsVectorSearch(vectors, numVecs, dim, query, topK);
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

// --- TS fallback helpers ---

function tsFallbackSum(f64: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < f64.length; i++) sum += f64[i];
  return sum;
}

function tsFallbackMin(f64: Float64Array): number {
  let min = Infinity;
  for (let i = 0; i < f64.length; i++) {
    if (f64[i] < min) min = f64[i];
  }
  return min;
}

function tsFallbackMax(f64: Float64Array): number {
  let max = -Infinity;
  for (let i = 0; i < f64.length; i++) {
    if (f64[i] > max) max = f64[i];
  }
  return max;
}

/** Cosine similarity vector search with top-K max-heap. */
function tsVectorSearch(
  vectors: Float32Array,
  numVecs: number,
  dim: number,
  query: Float32Array,
  topK: number,
): { indices: Uint32Array; scores: Float32Array } {
  // Precompute query norm
  let qNorm = 0;
  for (let d = 0; d < dim; d++) qNorm += query[d] * query[d];
  qNorm = Math.sqrt(qNorm);

  if (qNorm === 0 || numVecs === 0) {
    return { indices: new Uint32Array(0), scores: new Float32Array(0) };
  }

  // Max-heap on cosine distance (largest distance at top = worst result)
  const heap: { dist: number; idx: number }[] = [];

  for (let i = 0; i < numVecs; i++) {
    const base = i * dim;
    let dot = 0;
    let eNorm = 0;
    for (let d = 0; d < dim; d++) {
      const v = vectors[base + d];
      dot += v * query[d];
      eNorm += v * v;
    }
    const sim = eNorm > 0 ? dot / (qNorm * Math.sqrt(eNorm)) : 0;
    const dist = 1 - sim;

    if (heap.length < topK) {
      heap.push({ dist, idx: i });
      // Sift up
      let j = heap.length - 1;
      while (j > 0) {
        const p = (j - 1) >> 1;
        if (heap[j].dist > heap[p].dist) {
          [heap[j], heap[p]] = [heap[p], heap[j]];
          j = p;
        } else break;
      }
    } else if (dist < heap[0].dist) {
      heap[0] = { dist, idx: i };
      // Sift down
      let j = 0;
      while (true) {
        let t = j;
        const l = 2 * j + 1;
        const r = 2 * j + 2;
        if (l < heap.length && heap[l].dist > heap[t].dist) t = l;
        if (r < heap.length && heap[r].dist > heap[t].dist) t = r;
        if (t === j) break;
        [heap[j], heap[t]] = [heap[t], heap[j]];
        j = t;
      }
    }
  }

  // Sort by ascending distance (best first)
  heap.sort((a, b) => a.dist - b.dist);

  const indices = new Uint32Array(heap.length);
  const scores = new Float32Array(heap.length);
  for (let i = 0; i < heap.length; i++) {
    indices[i] = heap[i].idx;
    scores[i] = 1 - heap[i].dist; // cosine similarity score
  }

  return { indices, scores };
}
