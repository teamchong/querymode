/**
 * WASM integration tests — real WASM execution, no mocks.
 * Requires edgeq.wasm binary at src/wasm/edgeq.wasm.
 * Skips gracefully if binary not found.
 */
import { describe, it, expect, beforeAll } from "vitest";
import { instantiateWasm, type WasmEngine } from "./wasm-engine.js";
import type { QueryDescriptor } from "./client.js";
import type { PageInfo } from "./types.js";
import * as fs from "node:fs/promises";
import * as path from "node:path";

let wasm: WasmEngine;
let hasWasm = false;

beforeAll(async () => {
  try {
    const wasmPath = path.join(import.meta.dirname ?? ".", "wasm", "edgeq.wasm");
    const wasmBytes = await fs.readFile(wasmPath);
    const mod = await WebAssembly.compile(wasmBytes);
    wasm = await instantiateWasm(mod);
    hasWasm = true;
  } catch {
    hasWasm = false;
  }
});

describe.skipIf(!hasWasm)("WASM integration", () => {
  describe("int64 + float64 columns → SQL → rows", () => {
    it("registers columns and executes SELECT *", () => {
      wasm.exports.resetHeap();

      const table = "test_nums";
      const rowCount = 3;

      // Build int64 column: [10, 20, 30]
      const i64Buf = new BigInt64Array([10n, 20n, 30n]);
      const i64Pages = [i64Buf.buffer.slice(0)];
      const i64PageInfos: PageInfo[] = [{ byteOffset: 0n, byteLength: 24, rowCount: 3, nullCount: 0 }];
      expect(wasm.registerColumn(table, "id", "int64", i64Pages, i64PageInfos)).toBe(true);

      // Build float64 column: [1.5, 2.5, 3.5]
      const f64Buf = new Float64Array([1.5, 2.5, 3.5]);
      const f64Pages = [f64Buf.buffer.slice(0)];
      const f64PageInfos: PageInfo[] = [{ byteOffset: 0n, byteLength: 24, rowCount: 3, nullCount: 0 }];
      expect(wasm.registerColumn(table, "value", "float64", f64Pages, f64PageInfos)).toBe(true);

      // Execute SQL
      const query: QueryDescriptor = {
        table, filters: [], projections: [],
      };
      const rows = wasm.executeQuery(query);
      expect(rows).not.toBeNull();
      expect(rows!.length).toBe(rowCount);
      expect(rows![0].id).toBe(10n);
      expect(rows![0].value).toBe(1.5);
      expect(rows![1].id).toBe(20n);
      expect(rows![2].value).toBe(3.5);

      wasm.clearTable(table);
    });
  });

  describe("string column registration", () => {
    it("registers utf8 strings with length-prefixed encoding", () => {
      wasm.exports.resetHeap();
      const table = "test_strings";

      // Build length-prefixed string data: "hello" (5), "world" (5)
      const enc = new TextEncoder();
      const s1 = enc.encode("hello");
      const s2 = enc.encode("world");
      const buf = new ArrayBuffer(4 + s1.length + 4 + s2.length);
      const view = new DataView(buf);
      let off = 0;
      view.setUint32(off, s1.length, true); off += 4;
      new Uint8Array(buf, off, s1.length).set(s1); off += s1.length;
      view.setUint32(off, s2.length, true); off += 4;
      new Uint8Array(buf, off, s2.length).set(s2);

      const pageInfos: PageInfo[] = [{ byteOffset: 0n, byteLength: buf.byteLength, rowCount: 2, nullCount: 0 }];
      expect(wasm.registerColumn(table, "name", "utf8", [buf], pageInfos)).toBe(true);

      const query: QueryDescriptor = { table, filters: [], projections: [] };
      const rows = wasm.executeQuery(query);
      expect(rows).not.toBeNull();
      expect(rows!.length).toBe(2);
      expect(rows![0].name).toBe("hello");
      expect(rows![1].name).toBe("world");

      wasm.clearTable(table);
    });
  });

  describe("type promotion", () => {
    it("promotes int32 to int64 for WASM registration", () => {
      wasm.exports.resetHeap();
      const table = "test_promo";

      // int32 values
      const i32Buf = new Int32Array([100, 200, 300]);
      const pages = [i32Buf.buffer.slice(0)];
      const pageInfos: PageInfo[] = [{ byteOffset: 0n, byteLength: 12, rowCount: 3, nullCount: 0 }];
      expect(wasm.registerColumn(table, "val", "int32", pages, pageInfos)).toBe(true);

      const query: QueryDescriptor = { table, filters: [], projections: [] };
      const rows = wasm.executeQuery(query);
      expect(rows).not.toBeNull();
      expect(rows!.length).toBe(3);
      // Promoted to BigInt (int64)
      expect(rows![0].val).toBe(100n);
      expect(rows![2].val).toBe(300n);

      wasm.clearTable(table);
    });
  });

  describe("filter pushdown SQL", () => {
    it("filters with WHERE gt", () => {
      wasm.exports.resetHeap();
      const table = "test_filter";

      const i64Buf = new BigInt64Array([10n, 20n, 30n, 40n, 50n]);
      const pages = [i64Buf.buffer.slice(0)];
      const pageInfos: PageInfo[] = [{ byteOffset: 0n, byteLength: 40, rowCount: 5, nullCount: 0 }];
      expect(wasm.registerColumn(table, "score", "int64", pages, pageInfos)).toBe(true);

      const query: QueryDescriptor = {
        table, filters: [{ column: "score", op: "gt", value: 25 }], projections: [],
      };
      const rows = wasm.executeQuery(query);
      expect(rows).not.toBeNull();
      expect(rows!.length).toBe(3);
      expect(rows![0].score).toBe(30n);
      expect(rows![1].score).toBe(40n);
      expect(rows![2].score).toBe(50n);

      wasm.clearTable(table);
    });

    it("filters with WHERE eq", () => {
      wasm.exports.resetHeap();
      const table = "test_eq";

      const f64Buf = new Float64Array([1.0, 2.0, 3.0, 2.0]);
      const pages = [f64Buf.buffer.slice(0)];
      const pageInfos: PageInfo[] = [{ byteOffset: 0n, byteLength: 32, rowCount: 4, nullCount: 0 }];
      expect(wasm.registerColumn(table, "x", "float64", pages, pageInfos)).toBe(true);

      const query: QueryDescriptor = {
        table, filters: [{ column: "x", op: "eq", value: 2.0 }], projections: [],
      };
      const rows = wasm.executeQuery(query);
      expect(rows).not.toBeNull();
      expect(rows!.length).toBe(2);
      expect(rows![0].x).toBe(2.0);
      expect(rows![1].x).toBe(2.0);

      wasm.clearTable(table);
    });

    it("filters with WHERE lt on float64", () => {
      wasm.exports.resetHeap();
      const table = "test_lt";

      const f64Buf = new Float64Array([1.0, 2.0, 3.0, 4.0]);
      const pages = [f64Buf.buffer.slice(0)];
      const pageInfos: PageInfo[] = [{ byteOffset: 0n, byteLength: 32, rowCount: 4, nullCount: 0 }];
      expect(wasm.registerColumn(table, "val", "float64", pages, pageInfos)).toBe(true);

      const query: QueryDescriptor = {
        table, filters: [{ column: "val", op: "lt", value: 3.0 }], projections: [],
      };
      const rows = wasm.executeQuery(query);
      expect(rows).not.toBeNull();
      expect(rows!.length).toBe(2);

      wasm.clearTable(table);
    });
  });

  describe("vector search", () => {
    it("finds top-K nearest vectors via vectorSearchBuffer", () => {
      wasm.exports.resetHeap();
      const dim = 4;
      const numVectors = 5;

      // 5 vectors of dim 4
      const vectors = new Float32Array([
        1, 0, 0, 0,   // v0
        0, 1, 0, 0,   // v1
        0, 0, 1, 0,   // v2
        0, 0, 0, 1,   // v3
        0.9, 0.1, 0, 0, // v4 — closest to query
      ]);
      const query = new Float32Array([1, 0, 0, 0]);

      const result = wasm.vectorSearchBuffer(vectors, numVectors, dim, query, 2);
      expect(result.indices.length).toBe(2);
      expect(result.scores.length).toBe(2);
      // v0 and v4 should be the closest to [1,0,0,0]
      const topIndices = Array.from(result.indices);
      expect(topIndices).toContain(0);
      expect(topIndices).toContain(4);
    });
  });

  describe("buffer pool cache", () => {
    it("round-trips cacheSet → cacheGet", () => {
      const key = "test:page:0:100";
      const data = new Uint8Array([1, 2, 3, 4, 5]).buffer;

      const setOk = wasm.cacheSet(key, data);
      expect(setOk).toBe(true);

      expect(wasm.cacheHas(key)).toBe(true);

      const retrieved = wasm.cacheGet(key);
      expect(retrieved).not.toBeNull();
      expect(new Uint8Array(retrieved!)).toEqual(new Uint8Array(data));

      wasm.cacheClear();
      expect(wasm.cacheHas(key)).toBe(false);
    });

    it("returns null on cache miss", () => {
      const result = wasm.cacheGet("nonexistent:key");
      expect(result).toBeNull();
    });
  });

  describe("LIMIT and ORDER BY", () => {
    it("applies LIMIT to results", () => {
      wasm.exports.resetHeap();
      const table = "test_limit";

      const i64Buf = new BigInt64Array([1n, 2n, 3n, 4n, 5n]);
      const pages = [i64Buf.buffer.slice(0)];
      const pageInfos: PageInfo[] = [{ byteOffset: 0n, byteLength: 40, rowCount: 5, nullCount: 0 }];
      expect(wasm.registerColumn(table, "id", "int64", pages, pageInfos)).toBe(true);

      const query: QueryDescriptor = {
        table, filters: [], projections: [], limit: 2,
      };
      const rows = wasm.executeQuery(query);
      expect(rows).not.toBeNull();
      expect(rows!.length).toBe(2);

      wasm.clearTable(table);
    });
  });

  describe("multi-column SELECT with projections", () => {
    it("returns only projected columns", () => {
      wasm.exports.resetHeap();
      const table = "test_proj";

      const i64Buf = new BigInt64Array([1n, 2n, 3n]);
      const f64Buf = new Float64Array([10.0, 20.0, 30.0]);
      const i64PageInfos: PageInfo[] = [{ byteOffset: 0n, byteLength: 24, rowCount: 3, nullCount: 0 }];
      const f64PageInfos: PageInfo[] = [{ byteOffset: 0n, byteLength: 24, rowCount: 3, nullCount: 0 }];

      wasm.registerColumn(table, "id", "int64", [i64Buf.buffer.slice(0)], i64PageInfos);
      wasm.registerColumn(table, "score", "float64", [f64Buf.buffer.slice(0)], f64PageInfos);

      const query: QueryDescriptor = {
        table, filters: [], projections: ["score"],
      };
      const rows = wasm.executeQuery(query);
      expect(rows).not.toBeNull();
      expect(rows!.length).toBe(3);
      // Should have score but not id (WASM SQL executor returns only registered columns)
      expect(rows![0].score).toBe(10.0);

      wasm.clearTable(table);
    });
  });

  describe("bool column", () => {
    it("registers and queries bool values", () => {
      wasm.exports.resetHeap();
      const table = "test_bool";

      // Bool bitmap: [true, false, true] = 0b101 = 0x05
      const boolBuf = new Uint8Array([0x05]).buffer;
      const pageInfos: PageInfo[] = [{ byteOffset: 0n, byteLength: 1, rowCount: 3, nullCount: 0 }];
      expect(wasm.registerColumn(table, "flag", "bool", [boolBuf], pageInfos)).toBe(true);

      const query: QueryDescriptor = { table, filters: [], projections: [] };
      const rows = wasm.executeQuery(query);
      expect(rows).not.toBeNull();
      expect(rows!.length).toBe(3);
      // Bool is promoted to int64 (0/1)
      expect(rows![0].flag).toBe(1n);
      expect(rows![1].flag).toBe(0n);
      expect(rows![2].flag).toBe(1n);

      wasm.clearTable(table);
    });
  });

  describe("aggregates", () => {
    it("computes SUM via WASM", () => {
      wasm.exports.resetHeap();
      const table = "test_sum";

      const f64Buf = new Float64Array([10.0, 20.0, 30.0]);
      const pageInfos: PageInfo[] = [{ byteOffset: 0n, byteLength: 24, rowCount: 3, nullCount: 0 }];
      wasm.registerColumn(table, "amount", "float64", [f64Buf.buffer.slice(0)], pageInfos);

      const query: QueryDescriptor = {
        table, filters: [], projections: [],
        aggregates: [{ fn: "sum", column: "amount", alias: "total" }],
      };
      const rows = wasm.executeQuery(query);
      expect(rows).not.toBeNull();
      expect(rows!.length).toBe(1);
      expect(rows![0].total).toBeCloseTo(60.0);

      wasm.clearTable(table);
    });
  });

  describe("SIMD buffer aggregates", () => {
    it("sumFloat64 computes correct sum", () => {
      const data = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
      expect(wasm.sumFloat64(data.buffer)).toBeCloseTo(15.0);
    });

    it("minFloat64 finds minimum", () => {
      const data = new Float64Array([5.0, 2.0, 8.0, 1.0, 3.0]);
      expect(wasm.minFloat64(data.buffer)).toBeCloseTo(1.0);
    });

    it("maxFloat64 finds maximum", () => {
      const data = new Float64Array([5.0, 2.0, 8.0, 1.0, 3.0]);
      expect(wasm.maxFloat64(data.buffer)).toBeCloseTo(8.0);
    });

    it("avgFloat64 computes average", () => {
      const data = new Float64Array([10.0, 20.0, 30.0]);
      expect(wasm.avgFloat64(data.buffer)).toBeCloseTo(20.0);
    });
  });
});
