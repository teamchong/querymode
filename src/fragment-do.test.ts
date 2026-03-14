import { describe, it, expect, vi } from "vitest";
import { env, runInDurableObject } from "cloudflare:test";
import type { FragmentDO } from "./fragment-do.js";

// Mock WASM module import (not available in test worker)
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
    registerColumns: () => true,
    registerDecodedColumns: () => true,
    executeQuery: () => [],
    executeQueryColumnar: () => null,
    vectorSearchBuffer: () => ({ indices: new Uint32Array(0), scores: new Float32Array(0) }),
    cacheGet: () => null,
    cacheSet: () => false,
    cacheClear: () => {},
    cacheStats: () => ({ count: 0, bytes: 0, maxBytes: 0 }),
  }),
}));

declare module "cloudflare:test" {
  interface ProvidedEnv {
    FRAGMENT_DO: DurableObjectNamespace<FragmentDO>;
    DATA_BUCKET: R2Bucket;
  }
}

function getDoHandle() {
  const id = env.FRAGMENT_DO.idFromName("test-fragment");
  return env.FRAGMENT_DO.get(id);
}

describe("FragmentDO", () => {
  it("is constructible via real DO binding", async () => {
    const handle = getDoHandle();
    const result = await runInDurableObject(handle, (instance) => {
      return instance !== undefined;
    });
    expect(result).toBe(true);
  });

  it("returns valid result for scanRpc with empty fragments", async () => {
    const handle = getDoHandle();
    const result = await runInDurableObject(handle, async (instance) => {
      return instance.scanRpc(
        [],
        { table: "test", filters: [], projections: [] } as any,
      );
    });
    expect(result.rows).toEqual([]);
    expect(result.rowCount).toBe(0);
    expect(result.bytesRead).toBe(0);
    expect(result.pagesSkipped).toBe(0);
    expect(typeof result.durationMs).toBe("number");
  });

  it("handles scan with mock fragments gracefully when R2 returns null", async () => {
    const handle = getDoHandle();
    const result = await runInDurableObject(handle, async (instance) => {
      const meta = {
        name: "users",
        footer: {
          columnMetaStart: 0n, columnMetaOffsetsStart: 0n,
          globalBuffOffsetsStart: 0n, numGlobalBuffers: 0,
          numColumns: 1, majorVersion: 2, minorVersion: 0,
        },
        columns: [{
          name: "id",
          dtype: "int32" as const,
          pages: [{ byteOffset: 0n, byteLength: 12, rowCount: 3, nullCount: 0 }],
          nullCount: 0,
        }],
        totalRows: 3,
        fileSize: 100n,
        r2Key: "users.lance",
        updatedAt: Date.now(),
      };
      return instance.scanRpc(
        [{ r2Key: "users.lance", meta }],
        { table: "users", filters: [], projections: [] } as any,
      );
    });
    expect(result.rows).toEqual([]);
    expect(result.rowCount).toBe(0);
    expect(result.bytesRead).toBe(0);
    expect(result.columns).toEqual(["id"]);
  });
});
