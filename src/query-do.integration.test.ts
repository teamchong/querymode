/**
 * Query DO integration tests — real WASM but mocked R2.
 * Verifies full query flow: HTTP request → column registration → SQL → result rows.
 * Skips gracefully if WASM binary not found.
 */
import { describe, it, expect, vi, beforeAll, beforeEach } from "vitest";
import type { TableMeta, ColumnMeta, QueryResult, Row } from "./types.js";
import { bigIntReplacer } from "./decode.js";

let hasWasm = false;

beforeAll(async () => {
  try {
    const fs = await import("node:fs/promises");
    const path = await import("node:path");
    const wasmPath = path.join(import.meta.dirname ?? ".", "wasm", "querymode.wasm");
    await fs.stat(wasmPath);
    hasWasm = true;
  } catch {
    hasWasm = false;
  }
});

// Build mock R2 that returns column page data for Range reads
function buildMockR2(
  pageData: Map<string, { offset: number; data: ArrayBuffer }[]>,
  heads: Map<string, { size: number }>,
) {
  return {
    get: async (key: string, opts?: { range?: { offset: number; length: number } }) => {
      const pages = pageData.get(key);
      if (!pages) return null;

      if (opts?.range) {
        // Find matching page by offset
        const { offset, length } = opts.range;
        for (const page of pages) {
          if (page.offset <= offset && page.offset + page.data.byteLength >= offset + length) {
            const start = offset - page.offset;
            const slice = page.data.slice(start, start + length);
            return { arrayBuffer: async () => slice, text: async () => new TextDecoder().decode(new Uint8Array(slice)) };
          }
        }
        // Return full data if range covers start
        for (const page of pages) {
          return { arrayBuffer: async () => page.data, text: async () => "" };
        }
        return null;
      }

      // Return first page data as full file
      if (pages.length > 0) {
        return { arrayBuffer: async () => pages[0].data, text: async () => "" };
      }
      return null;
    },
    head: async (key: string) => heads.get(key) ?? null,
    list: async () => ({ objects: [] }),
    put: async () => null,
  };
}

describe.skipIf(!hasWasm)("Query DO integration (real WASM, mocked R2)", () => {
  it("handles query with int64 + float64 columns via WASM SQL executor", async () => {
    // Dynamically import to allow vi.mock to work
    const { QueryDO } = await import("./query-do.js");

    // Build column page data
    const idData = new BigInt64Array([1n, 2n, 3n]);
    const valueData = new Float64Array([10.5, 20.5, 30.5]);

    const cols: ColumnMeta[] = [
      {
        name: "id", dtype: "int64",
        pages: [{ byteOffset: 0n, byteLength: 24, rowCount: 3, nullCount: 0 }],
        nullCount: 0,
      },
      {
        name: "value", dtype: "float64",
        pages: [{ byteOffset: 100n, byteLength: 24, rowCount: 3, nullCount: 0 }],
        nullCount: 0,
      },
    ];

    const tableName = "test_table";
    const r2Key = "test_table.lance";

    // Mock R2 with page data
    const pageData = new Map<string, { offset: number; data: ArrayBuffer }[]>();
    pageData.set(r2Key, [
      { offset: 0, data: idData.buffer.slice(0) },
      { offset: 100, data: valueData.buffer.slice(0) },
    ]);
    const heads = new Map<string, { size: number }>();
    heads.set(r2Key, { size: 200 });

    const mockR2 = buildMockR2(pageData, heads);

    // Build pre-cached table meta
    const meta: TableMeta = {
      name: tableName,
      footer: {
        columnMetaStart: 0n, columnMetaOffsetsStart: 0n,
        globalBuffOffsetsStart: 0n, numGlobalBuffers: 0,
        numColumns: 2, majorVersion: 2, minorVersion: 0,
      },
      format: "lance",
      columns: cols,
      totalRows: 3,
      fileSize: 200n,
      r2Key,
      updatedAt: Date.now(),
    };

    // Load real WASM module
    const fs = await import("node:fs/promises");
    const path = await import("node:path");
    const wasmPath = path.join(import.meta.dirname ?? ".", "wasm", "querymode.wasm");
    const wasmBytes = await fs.readFile(wasmPath);
    const wasmModule = await WebAssembly.compile(wasmBytes);

    const mockState = {
      id: { toString: () => "test-query-do" },
      storage: {
        get: async (key: string) => {
          if (key === "region") return "test-region";
          return null;
        },
        put: async () => {},
        list: async () => {
          const map = new Map();
          map.set(`table:${tableName}`, meta);
          return map;
        },
      },
    } as any;

    const mockMasterDo = {
      fetch: async () => new Response(JSON.stringify({ registered: true, region: "test" })),
    };
    const mockMasterNs = {
      idFromName: () => "master-id",
      get: () => mockMasterDo,
    };

    const mockEnv = {
      DATA_BUCKET: mockR2,
      MASTER_DO: mockMasterNs,
      QUERY_DO: { idFromString: (id: string) => id, get: () => ({}) },
      FRAGMENT_DO: { idFromName: () => "frag-id", get: () => ({}) },
      QUERYMODE_WASM: wasmModule,
    } as any;

    const qdo = new QueryDO(mockState, mockEnv);

    // Issue a query request
    const queryReq = new Request("http://internal/query", {
      method: "POST",
      body: JSON.stringify({
        table: tableName,
        filters: [],
        projections: [],
      }),
      headers: { "content-type": "application/json" },
    });

    const response = await qdo.fetch(queryReq);
    expect(response.status).toBe(200);

    const result = (await response.json()) as QueryResult;
    expect(result.rowCount).toBe(3);
    expect(result.rows.length).toBe(3);
    // Verify actual data values
    expect(result.rows[0].id).toBe("1"); // bigint serialized as string via bigIntReplacer
    expect(Number(result.rows[0].value)).toBeCloseTo(10.5);
    expect(Number(result.rows[2].value)).toBeCloseTo(30.5);
  });

  it("returns 404 for unknown path", async () => {
    const { QueryDO } = await import("./query-do.js");
    const fs = await import("node:fs/promises");
    const path = await import("node:path");
    const wasmPath = path.join(import.meta.dirname ?? ".", "wasm", "querymode.wasm");
    const wasmBytes = await fs.readFile(wasmPath);
    const wasmModule = await WebAssembly.compile(wasmBytes);

    const mockState = {
      id: { toString: () => "test-do" },
      storage: { get: async () => null, put: async () => {}, list: async () => new Map() },
    } as any;

    const mockMasterDo = {
      fetch: async () => new Response(JSON.stringify({ registered: true })),
    };

    const mockEnv = {
      DATA_BUCKET: { get: async () => null, head: async () => null, list: async () => ({ objects: [] }) },
      MASTER_DO: { idFromName: () => "id", get: () => mockMasterDo },
      QUERY_DO: { idFromString: () => "id", get: () => ({}) },
      FRAGMENT_DO: { idFromName: () => "id", get: () => ({}) },
      QUERYMODE_WASM: wasmModule,
    } as any;

    const qdo = new QueryDO(mockState, mockEnv);
    const res = await qdo.fetch(new Request("http://internal/unknown"));
    expect(res.status).toBe(404);
  });

  it("handles /tables endpoint", async () => {
    const { QueryDO } = await import("./query-do.js");
    const fs = await import("node:fs/promises");
    const path = await import("node:path");
    const wasmPath = path.join(import.meta.dirname ?? ".", "wasm", "querymode.wasm");
    const wasmBytes = await fs.readFile(wasmPath);
    const wasmModule = await WebAssembly.compile(wasmBytes);

    const mockState = {
      id: { toString: () => "test-do" },
      storage: { get: async () => null, put: async () => {}, list: async () => new Map() },
    } as any;

    const mockMasterDo = {
      fetch: async () => new Response(JSON.stringify({ registered: true })),
    };

    const mockEnv = {
      DATA_BUCKET: { get: async () => null, head: async () => null, list: async () => ({ objects: [] }) },
      MASTER_DO: { idFromName: () => "id", get: () => mockMasterDo },
      QUERY_DO: { idFromString: () => "id", get: () => ({}) },
      FRAGMENT_DO: { idFromName: () => "id", get: () => ({}) },
      QUERYMODE_WASM: wasmModule,
    } as any;

    const qdo = new QueryDO(mockState, mockEnv);
    const res = await qdo.fetch(new Request("http://internal/tables"));
    expect(res.status).toBe(200);
    const body = await res.json() as { tables: unknown[] };
    expect(body.tables).toEqual([]);
  });
});
