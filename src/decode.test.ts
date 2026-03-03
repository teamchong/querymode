import { describe, it, expect } from "vitest";
import { decodePage, assembleRows, canSkipPage, matchesFilter, bigIntReplacer } from "./decode.js";
import type { ColumnMeta, PageInfo } from "./types.js";
import type { QueryDescriptor } from "./client.js";

describe("decodePage", () => {
  it("decodes int32 page", () => {
    const buf = new ArrayBuffer(12);
    const view = new DataView(buf);
    view.setInt32(0, 100, true);
    view.setInt32(4, -50, true);
    view.setInt32(8, 0, true);

    const values = decodePage(buf, "int32");
    expect(values).toEqual([100, -50, 0]);
  });

  it("decodes uint64 page correctly (not signed)", () => {
    const buf = new ArrayBuffer(8);
    const view = new DataView(buf);
    // Set a value > 2^63 (would be negative as signed)
    view.setBigUint64(0, 18446744073709551615n, true); // max uint64

    const values = decodePage(buf, "uint64");
    expect(values).toHaveLength(1);
    // FIX: This should be positive (getBigUint64), not negative (getBigInt64)
    expect(values[0]).toBe(18446744073709551615n);
    expect(values[0]! > 0n).toBe(true);
  });

  it("decodes float32 page", () => {
    const buf = new ArrayBuffer(8);
    const view = new DataView(buf);
    view.setFloat32(0, 3.14, true);
    view.setFloat32(4, -2.718, true);

    const values = decodePage(buf, "float32");
    expect(values).toHaveLength(2);
    expect(values[0]).toBeCloseTo(3.14, 2);
    expect(values[1]).toBeCloseTo(-2.718, 2);
  });

  it("decodes utf8 page", () => {
    const encoder = new TextEncoder();
    const str1 = encoder.encode("hello");
    const str2 = encoder.encode("world");

    const buf = new ArrayBuffer(4 + str1.length + 4 + str2.length);
    const view = new DataView(buf);
    let pos = 0;

    view.setUint32(pos, str1.length, true); pos += 4;
    new Uint8Array(buf, pos, str1.length).set(str1); pos += str1.length;

    view.setUint32(pos, str2.length, true); pos += 4;
    new Uint8Array(buf, pos, str2.length).set(str2);

    const values = decodePage(buf, "utf8");
    expect(values).toEqual(["hello", "world"]);
  });

  it("decodes float16 page", () => {
    const buf = new ArrayBuffer(2);
    const view = new DataView(buf);
    // Encode 1.0 as float16: sign=0, exp=15, mantissa=0 → 0x3C00
    view.setUint16(0, 0x3c00, true);

    const values = decodePage(buf, "float16");
    expect(values).toHaveLength(1);
    expect(values[0]).toBeCloseTo(1.0, 3);
  });

  it("decodes int64 page", () => {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setBigInt64(0, 9007199254740993n, true); // > Number.MAX_SAFE_INTEGER
    view.setBigInt64(8, -42n, true);

    const values = decodePage(buf, "int64");
    expect(values).toEqual([9007199254740993n, -42n]);
  });
});

describe("canSkipPage", () => {
  const page: PageInfo = {
    byteOffset: 0n,
    byteLength: 100,
    rowCount: 50,
    nullCount: 0,
    minValue: 10,
    maxValue: 90,
  };

  it("skips page where all values < filter gt threshold", () => {
    expect(canSkipPage(page, [{ column: "x", op: "gt", value: 100 }], "x")).toBe(true);
  });

  it("does not skip page that overlaps with gt filter", () => {
    expect(canSkipPage(page, [{ column: "x", op: "gt", value: 50 }], "x")).toBe(false);
  });

  it("skips page where eq value is outside range", () => {
    expect(canSkipPage(page, [{ column: "x", op: "eq", value: 5 }], "x")).toBe(true);
    expect(canSkipPage(page, [{ column: "x", op: "eq", value: 95 }], "x")).toBe(true);
  });

  it("does not skip page where eq value is in range", () => {
    expect(canSkipPage(page, [{ column: "x", op: "eq", value: 50 }], "x")).toBe(false);
  });

  it("skips page where all values >= filter lt threshold", () => {
    expect(canSkipPage(page, [{ column: "x", op: "lt", value: 10 }], "x")).toBe(true);
  });

  it("ignores filters for other columns", () => {
    expect(canSkipPage(page, [{ column: "y", op: "gt", value: 1000 }], "x")).toBe(false);
  });

  it("does not skip when page has no min/max stats", () => {
    const noStats: PageInfo = { byteOffset: 0n, byteLength: 100, rowCount: 50, nullCount: 0 };
    expect(canSkipPage(noStats, [{ column: "x", op: "gt", value: 100 }], "x")).toBe(false);
  });
});

describe("assembleRows", () => {
  function makeColumnData(name: string, values: number[]): [string, ArrayBuffer[]] {
    const buf = new ArrayBuffer(values.length * 4);
    const view = new DataView(buf);
    for (let i = 0; i < values.length; i++) {
      view.setInt32(i * 4, values[i], true);
    }
    return [name, [buf]];
  }

  const cols: ColumnMeta[] = [
    { name: "id", dtype: "int32", pages: [{ byteOffset: 0n, byteLength: 12, rowCount: 3, nullCount: 0 }], nullCount: 0 },
    { name: "score", dtype: "int32", pages: [{ byteOffset: 0n, byteLength: 12, rowCount: 3, nullCount: 0 }], nullCount: 0 },
  ];

  it("assembles rows from column data", () => {
    const columnData = new Map([
      makeColumnData("id", [1, 2, 3]),
      makeColumnData("score", [90, 80, 70]),
    ]);

    const query: QueryDescriptor = {
      table: "test",
      filters: [],
      projections: [],
    };

    const rows = assembleRows(columnData, cols, query);
    expect(rows).toHaveLength(3);
    expect(rows[0]).toEqual({ id: 1, score: 90 });
    expect(rows[2]).toEqual({ id: 3, score: 70 });
  });

  it("applies in-memory filters", () => {
    const columnData = new Map([
      makeColumnData("id", [1, 2, 3]),
      makeColumnData("score", [90, 80, 70]),
    ]);

    const query: QueryDescriptor = {
      table: "test",
      filters: [{ column: "score", op: "gt", value: 75 }],
      projections: [],
    };

    const rows = assembleRows(columnData, cols, query);
    expect(rows).toHaveLength(2);
    expect(rows[0].score).toBe(90);
    expect(rows[1].score).toBe(80);
  });

  it("applies limit without sort (early termination)", () => {
    const columnData = new Map([
      makeColumnData("id", [1, 2, 3]),
      makeColumnData("score", [90, 80, 70]),
    ]);

    const query: QueryDescriptor = {
      table: "test",
      filters: [],
      projections: [],
      limit: 2,
    };

    const rows = assembleRows(columnData, cols, query);
    expect(rows).toHaveLength(2);
    expect(rows[0].id).toBe(1);
    expect(rows[1].id).toBe(2);
  });

  it("applies sort + limit with top-K", () => {
    const ids = [1, 2, 3, 4, 5];
    const scores = [30, 50, 10, 40, 20];
    const columnData = new Map([
      makeColumnData("id", ids),
      makeColumnData("score", scores),
    ]);

    const bigCols: ColumnMeta[] = [
      { name: "id", dtype: "int32", pages: [{ byteOffset: 0n, byteLength: 20, rowCount: 5, nullCount: 0 }], nullCount: 0 },
      { name: "score", dtype: "int32", pages: [{ byteOffset: 0n, byteLength: 20, rowCount: 5, nullCount: 0 }], nullCount: 0 },
    ];

    const query: QueryDescriptor = {
      table: "test",
      filters: [],
      projections: [],
      sortColumn: "score",
      sortDirection: "desc",
      limit: 3,
    };

    const rows = assembleRows(columnData, bigCols, query);
    expect(rows).toHaveLength(3);
    expect(rows[0].score).toBe(50); // highest
    expect(rows[1].score).toBe(40);
    expect(rows[2].score).toBe(30);
  });

  it("handles fixed_size_list columns (embeddings)", () => {
    // 2 rows, dimension 3
    const embBuf = new ArrayBuffer(24);
    const embView = new DataView(embBuf);
    // Row 0: [1.0, 2.0, 3.0]
    embView.setFloat32(0, 1.0, true);
    embView.setFloat32(4, 2.0, true);
    embView.setFloat32(8, 3.0, true);
    // Row 1: [4.0, 5.0, 6.0]
    embView.setFloat32(12, 4.0, true);
    embView.setFloat32(16, 5.0, true);
    embView.setFloat32(20, 6.0, true);

    const embCols: ColumnMeta[] = [
      {
        name: "embedding",
        dtype: "fixed_size_list",
        listDimension: 3,
        pages: [{ byteOffset: 0n, byteLength: 24, rowCount: 2, nullCount: 0 }],
        nullCount: 0,
      },
    ];

    const columnData = new Map<string, ArrayBuffer[]>([["embedding", [embBuf]]]);

    const query: QueryDescriptor = {
      table: "test",
      filters: [],
      projections: [],
    };

    const rows = assembleRows(columnData, embCols, query);
    expect(rows).toHaveLength(2);

    // FIX: Each row should have its own Float32Array, not flattened floats
    const emb0 = rows[0].embedding as Float32Array;
    expect(emb0).toBeInstanceOf(Float32Array);
    expect(emb0.length).toBe(3);
    expect(emb0[0]).toBeCloseTo(1.0);
    expect(emb0[2]).toBeCloseTo(3.0);

    const emb1 = rows[1].embedding as Float32Array;
    expect(emb1[0]).toBeCloseTo(4.0);
    expect(emb1[2]).toBeCloseTo(6.0);
  });
});

describe("matchesFilter", () => {
  it("matches eq", () => {
    expect(matchesFilter(42, { column: "x", op: "eq", value: 42 })).toBe(true);
    expect(matchesFilter(43, { column: "x", op: "eq", value: 42 })).toBe(false);
  });

  it("matches gt/gte", () => {
    expect(matchesFilter(10, { column: "x", op: "gt", value: 5 })).toBe(true);
    expect(matchesFilter(5, { column: "x", op: "gt", value: 5 })).toBe(false);
    expect(matchesFilter(5, { column: "x", op: "gte", value: 5 })).toBe(true);
  });

  it("matches in", () => {
    expect(matchesFilter(2, { column: "x", op: "in", value: [1, 2, 3] })).toBe(true);
    expect(matchesFilter(4, { column: "x", op: "in", value: [1, 2, 3] })).toBe(false);
  });

  it("returns false for null", () => {
    expect(matchesFilter(null, { column: "x", op: "eq", value: 42 })).toBe(false);
  });
});

describe("bigIntReplacer", () => {
  it("converts BigInt to string", () => {
    expect(JSON.parse(JSON.stringify({ val: 42n }, bigIntReplacer))).toEqual({ val: "42" });
  });

  it("leaves non-BigInt values untouched", () => {
    expect(JSON.parse(JSON.stringify({ val: 42 }, bigIntReplacer))).toEqual({ val: 42 });
  });
});

describe("vector search", () => {
  it("throws without WASM engine", () => {
    const query: QueryDescriptor = {
      table: "test",
      filters: [],
      projections: [],
      vectorSearch: {
        column: "embedding",
        queryVector: new Float32Array([1, 0, 0, 0]),
        topK: 2,
      },
    };
    expect(() => assembleRows(new Map(), [], query)).toThrow("WASM engine required");
  });
});
