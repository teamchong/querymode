import { describe, it, expect, vi } from "vitest";
import { FragmentDO } from "./fragment-do.js";
import { bigIntReplacer } from "./decode.js";
import type { TableMeta, ColumnMeta } from "./types.js";

// Mock WASM module import (not available in vitest)
vi.mock("./wasm-module.js", () => ({ default: {} }));

// Mock instantiateWasm to return a minimal mock engine
vi.mock("./wasm-engine.js", () => ({
  instantiateWasm: async () => ({
    exports: {
      memory: new WebAssembly.Memory({ initial: 1 }),
      wasmAlloc: () => 0,
      resetHeap: () => {},
      resetResult: () => {},
      clearTable: () => {},
      clearTables: () => {},
    },
    reset: () => {},
    clearTable: () => {},
    registerColumn: () => true,
    executeQuery: () => [],
    vectorSearchBuffer: () => ({ indices: new Uint32Array(0), scores: new Float32Array(0) }),
    cacheGet: () => null,
    cacheSet: () => false,
    cacheClear: () => {},
    cacheStats: () => ({ count: 0, bytes: 0, maxBytes: 0 }),
  }),
}));

const mockState = {
  id: { toString: () => "test-fragment-id" },
  storage: { get: async () => null, put: async () => {}, list: async () => new Map() },
} as any;

const mockEnv = {
  DATA_BUCKET: { get: async () => null, head: async () => null },
  MASTER_DO: null,
  QUERY_DO: null,
} as any;

function makeScanRequest(url: string, body: unknown): Request {
  return new Request(url, {
    method: "POST",
    body: JSON.stringify(body, bigIntReplacer),
    headers: { "content-type": "application/json" },
  });
}

describe("FragmentDO", () => {
  it("is constructible", () => {
    const fdo = new FragmentDO(mockState, mockEnv);
    expect(fdo).toBeDefined();
  });

  it("returns 404 for unknown path", async () => {
    const fdo = new FragmentDO(mockState, mockEnv);
    const res = await fdo.fetch(new Request("http://internal/unknown"));
    expect(res.status).toBe(404);
    expect(await res.text()).toBe("Not found");
  });

  it("returns valid JSON for /scan with empty fragments", async () => {
    const fdo = new FragmentDO(mockState, mockEnv);
    const req = makeScanRequest("http://internal/scan", {
      fragments: [],
      query: { table: "test", filters: [], projections: [] },
    });
    const res = await fdo.fetch(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.rows).toEqual([]);
    expect(body.rowCount).toBe(0);
    expect(body.bytesRead).toBe(0);
    expect(body.pagesSkipped).toBe(0);
    expect(typeof body.durationMs).toBe("number");
  });

  it("handles scan with mock fragments gracefully when R2 returns null", async () => {
    const cols: ColumnMeta[] = [
      {
        name: "id",
        dtype: "int32",
        pages: [{ byteOffset: 0n, byteLength: 12, rowCount: 3, nullCount: 0 }],
        nullCount: 0,
      },
    ];
    const meta: TableMeta = {
      name: "users",
      footer: {
        columnMetaStart: 0n, columnMetaOffsetsStart: 0n,
        globalBuffOffsetsStart: 0n, numGlobalBuffers: 0,
        numColumns: 1, majorVersion: 2, minorVersion: 0,
      },
      columns: cols,
      totalRows: 3,
      fileSize: 100n,
      r2Key: "users.lance",
      updatedAt: Date.now(),
    };

    const fdo = new FragmentDO(mockState, mockEnv);
    const req = makeScanRequest("http://internal/scan", {
      fragments: [{ r2Key: "users.lance", meta }],
      query: { table: "users", filters: [], projections: [] },
    });
    const res = await fdo.fetch(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    // R2 returns null so no data is read, but response is still valid
    expect(body.rows).toEqual([]);
    expect(body.rowCount).toBe(0);
    expect(body.bytesRead).toBe(0);
    expect(body.columns).toEqual(["id"]);
  });
});
