import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { parseParquetFooter, parquetMetaToTableMeta, getParquetFooterLength, detectFormat } from "./parquet.js";
import { decodeParquetColumnChunk, decompressSnappy, decodePlainValues } from "./parquet-decode.js";
import { canSkipPage, decodePage, matchesFilter } from "./decode.js";
import { parseFooter, parseColumnMetaFromProtobuf, FOOTER_SIZE } from "./footer.js";
import { parseManifest } from "./manifest.js";
import type { PageInfo, PageEncoding, DataType } from "./types.js";
import type { WasmEngine } from "./wasm-engine.js";

const FIXTURES = join(import.meta.dirname, "../wasm/tests/fixtures");
const GENERATED = join(FIXTURES, "generated");

// Minimal WASM mock -- only needed for Parquet decompression that uses WASM (ZSTD/GZIP/LZ4).
// Snappy is pure TS so it works without WASM.
const mockWasm = {} as WasmEngine;

// ============================================================================
// Helper: load a Parquet file's footer bytes
// ============================================================================

function loadParquetFooterBytes(filePath: string): { footerBuf: ArrayBuffer; fileSize: number } {
  const file = readFileSync(filePath);
  const fileSize = file.length;
  const last8 = file.buffer.slice(file.byteOffset + fileSize - 8, file.byteOffset + fileSize);
  const footerLen = getParquetFooterLength(last8);
  if (footerLen === null) throw new Error(`Not a valid Parquet file: ${filePath}`);
  const footerStart = fileSize - 8 - footerLen;
  const footerBuf = file.buffer.slice(file.byteOffset + footerStart, file.byteOffset + footerStart + footerLen);
  return { footerBuf, fileSize };
}

// ============================================================================
// 1. Parquet footer parsing correctness
// ============================================================================

describe("Parquet footer parsing correctness", () => {
  describe("simple_plain.parquet", () => {
    it("detects Parquet format from file tail", () => {
      const file = readFileSync(join(FIXTURES, "simple_plain.parquet"));
      const tail = file.buffer.slice(file.byteOffset + file.length - 8, file.byteOffset + file.length);
      expect(detectFormat(tail)).toBe("parquet");
    });

    it("parses footer with correct row count and columns", () => {
      const { footerBuf, fileSize } = loadParquetFooterBytes(join(FIXTURES, "simple_plain.parquet"));
      const meta = parseParquetFooter(footerBuf);
      expect(meta).not.toBe(null);
      expect(meta!.numRows).toBeGreaterThan(0);
      expect(meta!.version).toBeGreaterThanOrEqual(1);
      expect(meta!.schema.length).toBeGreaterThan(1); // root + at least 1 leaf
      expect(meta!.rowGroups.length).toBeGreaterThanOrEqual(1);
    });

    it("converts to TableMeta with correct column types", () => {
      const { footerBuf, fileSize } = loadParquetFooterBytes(join(FIXTURES, "simple_plain.parquet"));
      const meta = parseParquetFooter(footerBuf)!;
      const tableMeta = parquetMetaToTableMeta(meta, "simple_plain.parquet", BigInt(fileSize));
      expect(tableMeta.format).toBe("parquet");
      expect(tableMeta.totalRows).toBe(meta.numRows);
      expect(tableMeta.columns.length).toBeGreaterThan(0);
      // Every column should have pages equal to the number of row groups
      for (const col of tableMeta.columns) {
        expect(col.pages.length).toBe(meta.rowGroups.length);
      }
    });
  });

  describe("simple.parquet (Snappy compressed)", () => {
    it("parses footer successfully", () => {
      const { footerBuf } = loadParquetFooterBytes(join(FIXTURES, "simple.parquet"));
      const meta = parseParquetFooter(footerBuf);
      expect(meta).not.toBe(null);
      expect(meta!.numRows).toBeGreaterThan(0);
    });
  });

  describe("simple_snappy.parquet", () => {
    it("parses footer and detects Snappy compression", () => {
      const { footerBuf, fileSize } = loadParquetFooterBytes(join(FIXTURES, "simple_snappy.parquet"));
      const meta = parseParquetFooter(footerBuf)!;
      const tableMeta = parquetMetaToTableMeta(meta, "simple_snappy.parquet", BigInt(fileSize));

      // At least one column should use SNAPPY
      const hasSnappy = tableMeta.columns.some(col =>
        col.pages.some(p => p.encoding?.compression === "SNAPPY")
      );
      expect(hasSnappy).toBe(true);
    });
  });

  describe("generated multi-row-group Parquet files", () => {
    it("bench_100k_3col.parquet has multiple row groups with statistics", () => {
      const { footerBuf, fileSize } = loadParquetFooterBytes(join(GENERATED, "bench_100k_3col.parquet"));
      const meta = parseParquetFooter(footerBuf)!;
      expect(meta.numRows).toBe(100_000);
      expect(meta.rowGroups.length).toBe(10); // 100K / 10K per row group

      const tableMeta = parquetMetaToTableMeta(meta, "bench_100k_3col.parquet", BigInt(fileSize));
      expect(tableMeta.columns.length).toBe(3);

      // Verify column names
      const colNames = tableMeta.columns.map(c => c.name).sort();
      expect(colNames).toEqual(["category", "id", "value"]);

      // Verify data types
      const idCol = tableMeta.columns.find(c => c.name === "id")!;
      expect(idCol.dtype).toBe("int64");
      const valueCol = tableMeta.columns.find(c => c.name === "value")!;
      expect(valueCol.dtype).toBe("float64");
      const catCol = tableMeta.columns.find(c => c.name === "category")!;
      expect(catCol.dtype).toBe("utf8");

      // Each column should have 10 pages (one per row group)
      expect(idCol.pages.length).toBe(10);
      expect(valueCol.pages.length).toBe(10);
      expect(catCol.pages.length).toBe(10);

      // Statistics should be present on numeric columns
      for (const page of idCol.pages) {
        expect(page.minValue).toBeDefined();
        expect(page.maxValue).toBeDefined();
        expect(typeof page.minValue).toBe("number");
        expect(typeof page.maxValue).toBe("number");
      }
    });

    it("bench_100k_numeric.parquet has correct structure", () => {
      const { footerBuf, fileSize } = loadParquetFooterBytes(join(GENERATED, "bench_100k_numeric.parquet"));
      const meta = parseParquetFooter(footerBuf)!;
      expect(meta.numRows).toBe(100_000);
      expect(meta.rowGroups.length).toBe(10);

      const tableMeta = parquetMetaToTableMeta(meta, "bench_100k_numeric.parquet", BigInt(fileSize));
      expect(tableMeta.columns.length).toBe(2);
      expect(tableMeta.columns.map(c => c.name).sort()).toEqual(["id", "value"]);
    });

    it("bench_1m_numeric.parquet has 100 row groups", () => {
      const { footerBuf } = loadParquetFooterBytes(join(GENERATED, "bench_1m_numeric.parquet"));
      const meta = parseParquetFooter(footerBuf)!;
      expect(meta.numRows).toBe(1_000_000);
      expect(meta.rowGroups.length).toBe(100);
    });

    it("total rows across all row groups matches numRows", () => {
      const { footerBuf } = loadParquetFooterBytes(join(GENERATED, "bench_100k_3col.parquet"));
      const meta = parseParquetFooter(footerBuf)!;
      const sumFromRgs = meta.rowGroups.reduce((sum, rg) => sum + rg.numRows, 0);
      expect(sumFromRgs).toBe(meta.numRows);
    });
  });
});

// ============================================================================
// 2. Parquet column decoding correctness
// ============================================================================

describe("Parquet column decoding correctness", () => {
  describe("bench_100k_3col.parquet - INT64 decoding", () => {
    it("decodes id column from first row group with correct BigInt values", () => {
      const filePath = join(GENERATED, "bench_100k_3col.parquet");
      const file = readFileSync(filePath);
      const { footerBuf, fileSize } = loadParquetFooterBytes(filePath);
      const meta = parseParquetFooter(footerBuf)!;
      const tableMeta = parquetMetaToTableMeta(meta, "bench_100k_3col.parquet", BigInt(fileSize));

      const idCol = tableMeta.columns.find(c => c.name === "id")!;
      const firstPage = idCol.pages[0];
      const offset = Number(firstPage.byteOffset);
      const length = firstPage.byteLength;

      const chunkBuf = file.buffer.slice(file.byteOffset + offset, file.byteOffset + offset + length);
      const values = decodeParquetColumnChunk(chunkBuf, firstPage.encoding!, idCol.dtype, firstPage.rowCount, mockWasm);

      expect(values.length).toBe(10_000);
      // IDs are sequential starting from 0
      expect(values[0]).toBe(0n);
      expect(values[1]).toBe(1n);
      expect(values[9999]).toBe(9999n);
    });

    it("decodes id column from second row group with correct offset", () => {
      const filePath = join(GENERATED, "bench_100k_3col.parquet");
      const file = readFileSync(filePath);
      const { footerBuf, fileSize } = loadParquetFooterBytes(filePath);
      const meta = parseParquetFooter(footerBuf)!;
      const tableMeta = parquetMetaToTableMeta(meta, "bench_100k_3col.parquet", BigInt(fileSize));

      const idCol = tableMeta.columns.find(c => c.name === "id")!;
      const secondPage = idCol.pages[1];
      const offset = Number(secondPage.byteOffset);
      const length = secondPage.byteLength;

      const chunkBuf = file.buffer.slice(file.byteOffset + offset, file.byteOffset + offset + length);
      const values = decodeParquetColumnChunk(chunkBuf, secondPage.encoding!, idCol.dtype, secondPage.rowCount, mockWasm);

      expect(values.length).toBe(10_000);
      // Second row group should have IDs starting at 10000
      expect(values[0]).toBe(10000n);
      expect(values[9999]).toBe(19999n);
    });
  });

  describe("bench_100k_3col.parquet - FLOAT64 decoding", () => {
    it("decodes value column with finite float64 numbers", () => {
      const filePath = join(GENERATED, "bench_100k_3col.parquet");
      const file = readFileSync(filePath);
      const { footerBuf, fileSize } = loadParquetFooterBytes(filePath);
      const meta = parseParquetFooter(footerBuf)!;
      const tableMeta = parquetMetaToTableMeta(meta, "bench_100k_3col.parquet", BigInt(fileSize));

      const valueCol = tableMeta.columns.find(c => c.name === "value")!;
      const firstPage = valueCol.pages[0];
      const offset = Number(firstPage.byteOffset);
      const length = firstPage.byteLength;

      const chunkBuf = file.buffer.slice(file.byteOffset + offset, file.byteOffset + offset + length);
      const values = decodeParquetColumnChunk(chunkBuf, firstPage.encoding!, valueCol.dtype, firstPage.rowCount, mockWasm);

      expect(values.length).toBe(10_000);
      // All values should be finite numbers in range [0, 1000)
      for (const v of values) {
        expect(typeof v).toBe("number");
        expect(Number.isFinite(v as number)).toBe(true);
        expect(v as number).toBeGreaterThanOrEqual(0);
        expect(v as number).toBeLessThan(1000);
      }
    });
  });

  describe("bench_100k_3col.parquet - UTF8 decoding", () => {
    it("decodes category column with correct string values", () => {
      const filePath = join(GENERATED, "bench_100k_3col.parquet");
      const file = readFileSync(filePath);
      const { footerBuf, fileSize } = loadParquetFooterBytes(filePath);
      const meta = parseParquetFooter(footerBuf)!;
      const tableMeta = parquetMetaToTableMeta(meta, "bench_100k_3col.parquet", BigInt(fileSize));

      const catCol = tableMeta.columns.find(c => c.name === "category")!;
      const firstPage = catCol.pages[0];
      const offset = Number(firstPage.byteOffset);
      const length = firstPage.byteLength;

      const chunkBuf = file.buffer.slice(file.byteOffset + offset, file.byteOffset + offset + length);
      const values = decodeParquetColumnChunk(chunkBuf, firstPage.encoding!, catCol.dtype, firstPage.rowCount, mockWasm);

      expect(values.length).toBe(10_000);
      const expectedCategories = new Set(["alpha", "beta", "gamma", "delta", "epsilon"]);
      for (const v of values) {
        expect(typeof v).toBe("string");
        expect(expectedCategories.has(v as string)).toBe(true);
      }
      // First values cycle: alpha, beta, gamma, delta, epsilon, ...
      expect(values[0]).toBe("alpha");
      expect(values[1]).toBe("beta");
      expect(values[2]).toBe("gamma");
      expect(values[3]).toBe("delta");
      expect(values[4]).toBe("epsilon");
    });
  });

  describe("multi-row-group decoding completeness", () => {
    it("decoding all row groups yields the full row count", () => {
      const filePath = join(GENERATED, "bench_100k_numeric.parquet");
      const file = readFileSync(filePath);
      const { footerBuf, fileSize } = loadParquetFooterBytes(filePath);
      const meta = parseParquetFooter(footerBuf)!;
      const tableMeta = parquetMetaToTableMeta(meta, "bench_100k_numeric.parquet", BigInt(fileSize));

      const idCol = tableMeta.columns.find(c => c.name === "id")!;
      let totalDecoded = 0;
      for (const page of idCol.pages) {
        const offset = Number(page.byteOffset);
        const chunkBuf = file.buffer.slice(file.byteOffset + offset, file.byteOffset + offset + page.byteLength);
        const values = decodeParquetColumnChunk(chunkBuf, page.encoding!, idCol.dtype, page.rowCount, mockWasm);
        totalDecoded += values.length;
      }
      expect(totalDecoded).toBe(100_000);
    });
  });

  describe("Snappy compressed Parquet decoding", () => {
    it("decompresses and decodes data from simple_snappy.parquet", () => {
      const filePath = join(FIXTURES, "simple_snappy.parquet");
      const file = readFileSync(filePath);
      const { footerBuf, fileSize } = loadParquetFooterBytes(filePath);
      const meta = parseParquetFooter(footerBuf)!;
      const tableMeta = parquetMetaToTableMeta(meta, "simple_snappy.parquet", BigInt(fileSize));

      // Should have at least one column
      expect(tableMeta.columns.length).toBeGreaterThan(0);

      // Decode the first column's first page
      const col = tableMeta.columns[0];
      const page = col.pages[0];
      const offset = Number(page.byteOffset);
      const chunkBuf = file.buffer.slice(file.byteOffset + offset, file.byteOffset + offset + page.byteLength);
      const values = decodeParquetColumnChunk(chunkBuf, page.encoding!, col.dtype, page.rowCount, mockWasm);

      expect(values.length).toBeGreaterThan(0);
      expect(values.length).toBe(page.rowCount);
    });
  });

  describe("Snappy decompression round-trip", () => {
    it("Snappy-compressed file decodes to plausible values", () => {
      // simple_snappy.parquet uses dictionary+Snappy encoding.
      // Verify that decoded values are sensible (correct type, non-garbage).
      const snappyPath = join(FIXTURES, "simple_snappy.parquet");
      const snappyFile = readFileSync(snappyPath);
      const { footerBuf, fileSize } = loadParquetFooterBytes(snappyPath);
      const meta = parseParquetFooter(footerBuf)!;
      const tableMeta = parquetMetaToTableMeta(meta, "simple_snappy.parquet", BigInt(fileSize));

      // Decode each column and verify types match expectations
      for (const col of tableMeta.columns) {
        const page = col.pages[0];
        const offset = Number(page.byteOffset);
        const chunkBuf = snappyFile.buffer.slice(snappyFile.byteOffset + offset, snappyFile.byteOffset + offset + page.byteLength);
        const values = decodeParquetColumnChunk(chunkBuf, page.encoding!, col.dtype, page.rowCount, mockWasm);

        expect(values.length).toBe(page.rowCount);

        if (col.dtype === "int64") {
          for (const v of values) expect(typeof v).toBe("bigint");
        } else if (col.dtype === "float64") {
          for (const v of values) {
            expect(typeof v).toBe("number");
            expect(Number.isFinite(v as number)).toBe(true);
          }
        } else if (col.dtype === "utf8") {
          for (const v of values) expect(typeof v).toBe("string");
        }
      }
    });

    it("decompressSnappy produces correct output for known input", () => {
      // Snappy-compress a small buffer manually and verify round-trip.
      // Encode: literal "hello" → tag byte for 5-byte literal: (4 << 2) | 0 = 16, then data
      const literalTag = (4 << 2) | 0; // length-1=4, shifted, type=literal
      const data = new TextEncoder().encode("hello");
      // Snappy format: uncompressed length varint (5), then literal tag + data
      const compressed = new Uint8Array([5, literalTag, ...data]);
      const result = decompressSnappy(compressed);
      expect(new TextDecoder().decode(result)).toBe("hello");
    });
  });
});

// ============================================================================
// 3. Page skipping correctness
// ============================================================================

describe("Page skipping correctness", () => {
  const pageWithStats: PageInfo = {
    byteOffset: 0n,
    byteLength: 1000,
    rowCount: 100,
    nullCount: 0,
    minValue: 100,
    maxValue: 500,
  };

  const pageWithoutStats: PageInfo = {
    byteOffset: 0n,
    byteLength: 1000,
    rowCount: 100,
    nullCount: 0,
    // No minValue/maxValue
  };

  const pageWithStringStats: PageInfo = {
    byteOffset: 0n,
    byteLength: 1000,
    rowCount: 100,
    nullCount: 0,
    minValue: "apple",
    maxValue: "pear",
  };

  describe("gt operator", () => {
    it("skips when max <= threshold", () => {
      expect(canSkipPage(pageWithStats, [{ column: "x", op: "gt", value: 500 }], "x")).toBe(true);
      expect(canSkipPage(pageWithStats, [{ column: "x", op: "gt", value: 600 }], "x")).toBe(true);
    });

    it("does not skip when max > threshold", () => {
      expect(canSkipPage(pageWithStats, [{ column: "x", op: "gt", value: 499 }], "x")).toBe(false);
      expect(canSkipPage(pageWithStats, [{ column: "x", op: "gt", value: 100 }], "x")).toBe(false);
    });
  });

  describe("gte operator", () => {
    it("skips when max < threshold", () => {
      expect(canSkipPage(pageWithStats, [{ column: "x", op: "gte", value: 501 }], "x")).toBe(true);
    });

    it("does not skip when max >= threshold", () => {
      expect(canSkipPage(pageWithStats, [{ column: "x", op: "gte", value: 500 }], "x")).toBe(false);
    });
  });

  describe("lt operator", () => {
    it("skips when min >= threshold", () => {
      expect(canSkipPage(pageWithStats, [{ column: "x", op: "lt", value: 100 }], "x")).toBe(true);
      expect(canSkipPage(pageWithStats, [{ column: "x", op: "lt", value: 50 }], "x")).toBe(true);
    });

    it("does not skip when min < threshold", () => {
      expect(canSkipPage(pageWithStats, [{ column: "x", op: "lt", value: 101 }], "x")).toBe(false);
    });
  });

  describe("lte operator", () => {
    it("skips when min > threshold", () => {
      expect(canSkipPage(pageWithStats, [{ column: "x", op: "lte", value: 99 }], "x")).toBe(true);
    });

    it("does not skip when min <= threshold", () => {
      expect(canSkipPage(pageWithStats, [{ column: "x", op: "lte", value: 100 }], "x")).toBe(false);
    });
  });

  describe("eq operator", () => {
    it("skips when value is outside [min, max] range", () => {
      expect(canSkipPage(pageWithStats, [{ column: "x", op: "eq", value: 50 }], "x")).toBe(true);
      expect(canSkipPage(pageWithStats, [{ column: "x", op: "eq", value: 600 }], "x")).toBe(true);
    });

    it("does not skip when value is inside range", () => {
      expect(canSkipPage(pageWithStats, [{ column: "x", op: "eq", value: 300 }], "x")).toBe(false);
    });

    it("does not skip when value is at boundary", () => {
      expect(canSkipPage(pageWithStats, [{ column: "x", op: "eq", value: 100 }], "x")).toBe(false);
      expect(canSkipPage(pageWithStats, [{ column: "x", op: "eq", value: 500 }], "x")).toBe(false);
    });
  });

  describe("pages without stats are never skipped (conservative)", () => {
    it("does not skip for gt filter", () => {
      expect(canSkipPage(pageWithoutStats, [{ column: "x", op: "gt", value: 999999 }], "x")).toBe(false);
    });

    it("does not skip for eq filter", () => {
      expect(canSkipPage(pageWithoutStats, [{ column: "x", op: "eq", value: 999999 }], "x")).toBe(false);
    });

    it("does not skip for lt filter", () => {
      expect(canSkipPage(pageWithoutStats, [{ column: "x", op: "lt", value: -999999 }], "x")).toBe(false);
    });
  });

  describe("column name mismatch never triggers skip", () => {
    it("does not skip when filter column differs from page column", () => {
      expect(canSkipPage(pageWithStats, [{ column: "other", op: "gt", value: 999999 }], "x")).toBe(false);
    });
  });

  describe("string statistics work for skipping", () => {
    it("skips when eq value is outside string range", () => {
      expect(canSkipPage(pageWithStringStats, [{ column: "x", op: "eq", value: "zzz" }], "x")).toBe(true);
      expect(canSkipPage(pageWithStringStats, [{ column: "x", op: "eq", value: "aaa" }], "x")).toBe(true);
    });

    it("does not skip when eq value is in string range", () => {
      expect(canSkipPage(pageWithStringStats, [{ column: "x", op: "eq", value: "banana" }], "x")).toBe(false);
    });
  });

  describe("page skipping with real Parquet statistics", () => {
    it("skips row groups where id column max < filter threshold", () => {
      const filePath = join(GENERATED, "bench_100k_3col.parquet");
      const file = readFileSync(filePath);
      const { footerBuf, fileSize } = loadParquetFooterBytes(filePath);
      const meta = parseParquetFooter(footerBuf)!;
      const tableMeta = parquetMetaToTableMeta(meta, "bench_100k_3col.parquet", BigInt(fileSize));

      const idCol = tableMeta.columns.find(c => c.name === "id")!;
      // Filter: id > 95000. First 9 row groups (IDs 0-89999) should be skippable.
      const filter = { column: "id", op: "gt" as const, value: 95000 };
      let skipped = 0;
      let kept = 0;
      for (const page of idCol.pages) {
        if (canSkipPage(page, [filter], "id")) skipped++;
        else kept++;
      }
      // At least the first 9 row groups should be skipped (max values 9999 through 89999)
      expect(skipped).toBeGreaterThanOrEqual(9);
      expect(kept).toBeGreaterThanOrEqual(1);
    });
  });
});

// ============================================================================
// 4. Lance footer parsing
// ============================================================================

describe("Lance footer parsing", () => {
  // Helper to find the single .lance data file in a dataset's data/ directory
  function findLanceDataFile(datasetName: string): string {
    const dataDir = join(FIXTURES, datasetName, "data");
    const files = readdirSync(dataDir);
    const lanceFile = files.find(f => f.endsWith(".lance"));
    return join(dataDir, lanceFile!);
  }

  const lanceDatasets = [
    { name: "simple_int64.lance", numCols: 1 },
    { name: "simple_float64.lance", numCols: 1 },
    { name: "mixed_types.lance", numCols: 3 },
    { name: "basic_types.lance", numCols: null },
  ];

  for (const ds of lanceDatasets) {
    describe(ds.name, () => {
      it("detects Lance format from data file tail", () => {
        const file = readFileSync(findLanceDataFile(ds.name));
        const tail = file.buffer.slice(file.byteOffset + file.length - 8, file.byteOffset + file.length);
        expect(detectFormat(tail)).toBe("lance");
      });

      it("parses the 40-byte Lance footer with valid structure", () => {
        const file = readFileSync(findLanceDataFile(ds.name));
        const footer = parseFooter(file.buffer.slice(file.byteOffset, file.byteOffset + file.length));
        expect(footer).not.toBe(null);
        // These fixtures are Lance v0.3 format
        expect(footer!.majorVersion).toBeGreaterThanOrEqual(0);
        if (ds.numCols !== null) {
          expect(footer!.numColumns).toBe(ds.numCols);
        } else {
          expect(footer!.numColumns).toBeGreaterThan(0);
        }
        expect(Number(footer!.columnMetaStart)).toBeGreaterThan(0);
      });

      it("parses column metadata protobuf with correct column count", () => {
        const file = readFileSync(findLanceDataFile(ds.name));
        const fileBytes = file.buffer.slice(file.byteOffset, file.byteOffset + file.length);
        const footer = parseFooter(fileBytes)!;

        const metaStart = Number(footer.columnMetaStart);
        const metaEnd = file.length - FOOTER_SIZE;
        const metaBuf = fileBytes.slice(metaStart, metaEnd);
        const columns = parseColumnMetaFromProtobuf(metaBuf, footer.numColumns);

        // Column count from protobuf should match footer
        expect(columns.length).toBe(footer.numColumns);

        // Each column should have a name (may be default "column_N" for v0.x data files)
        for (const col of columns) {
          expect(col.name).toBeDefined();
          expect(col.name.length).toBeGreaterThan(0);
        }
      });
    });
  }

  describe("Lance manifest parsing", () => {
    it("parses simple_int64.lance manifest", () => {
      const manifestPath = join(FIXTURES, "simple_int64.lance", "_versions", "1.manifest");
      const file = readFileSync(manifestPath);
      const manifest = parseManifest(file.buffer.slice(file.byteOffset, file.byteOffset + file.length));
      expect(manifest).not.toBe(null);
      expect(manifest!.version).toBeGreaterThanOrEqual(1);
      expect(manifest!.fragments.length).toBeGreaterThanOrEqual(1);
      expect(manifest!.totalRows).toBe(5); // Expected from simple_int64.expected.json
      expect(manifest!.schema.length).toBeGreaterThan(0);
      expect(manifest!.schema.find(f => f.name === "id")).toBeDefined();
    });

    it("parses multiple_batches.lance manifest with 9 total rows", () => {
      // Try highest version manifest first
      const manifestPath = join(FIXTURES, "multiple_batches.lance", "_versions", "3.manifest");
      const file = readFileSync(manifestPath);
      const manifest = parseManifest(file.buffer.slice(file.byteOffset, file.byteOffset + file.length));
      expect(manifest).not.toBe(null);
      expect(manifest!.totalRows).toBe(9); // Expected from multiple_batches.expected.json
      expect(manifest!.fragments.length).toBe(3); // 3 batches
    });

    it("parses mixed_types.lance manifest with correct schema fields", () => {
      const manifestPath = join(FIXTURES, "mixed_types.lance", "_versions", "1.manifest");
      const file = readFileSync(manifestPath);
      const manifest = parseManifest(file.buffer.slice(file.byteOffset, file.byteOffset + file.length));
      expect(manifest).not.toBe(null);
      expect(manifest!.totalRows).toBe(3);
      const fieldNames = manifest!.schema.map(f => f.name);
      expect(fieldNames).toContain("id");
      expect(fieldNames).toContain("value");
      expect(fieldNames).toContain("name");
    });
  });

  describe("Lance manifest column names match expected.json", () => {
    it("simple_int64.lance manifest has 'id' column", () => {
      const manifestPath = join(FIXTURES, "simple_int64.lance", "_versions", "1.manifest");
      const file = readFileSync(manifestPath);
      const manifest = parseManifest(file.buffer.slice(file.byteOffset, file.byteOffset + file.length))!;
      const fieldNames = manifest.schema.map(f => f.name);
      expect(fieldNames).toContain("id");
    });

    it("simple_float64.lance manifest has 'value' column", () => {
      const manifestPath = join(FIXTURES, "simple_float64.lance", "_versions", "1.manifest");
      const file = readFileSync(manifestPath);
      const manifest = parseManifest(file.buffer.slice(file.byteOffset, file.byteOffset + file.length))!;
      const fieldNames = manifest.schema.map(f => f.name);
      expect(fieldNames).toContain("value");
    });
  });

  describe("Lance data file decoding against expected values", () => {
    it("decodes simple_int64.lance data correctly", () => {
      const file = readFileSync(findLanceDataFile("simple_int64.lance"));
      const fileBytes = file.buffer.slice(file.byteOffset, file.byteOffset + file.length);
      const footer = parseFooter(fileBytes)!;

      const metaStart = Number(footer.columnMetaStart);
      const metaEnd = file.length - FOOTER_SIZE;
      const columns = parseColumnMetaFromProtobuf(fileBytes.slice(metaStart, metaEnd), footer.numColumns);

      // In v0.x data files, column names are generic. Use first column (there's only one).
      expect(columns.length).toBe(1);
      const col = columns[0];

      // The manifest tells us this is int64, but the data file protobuf may encode
      // the type differently. Verify we can decode the raw page data.
      const page = col.pages[0];
      if (page && page.byteLength > 0) {
        const pageData = fileBytes.slice(Number(page.byteOffset), Number(page.byteOffset) + page.byteLength);
        // Decode as int64 (we know from expected.json this has 5 int64 values: [1,2,3,4,5])
        const values = decodePage(new Uint8Array(pageData).buffer, "int64", page.nullCount, page.rowCount);
        expect(values.length).toBe(5);
        expect(values).toEqual([1n, 2n, 3n, 4n, 5n]);
      }
    });

    it("decodes simple_float64.lance data correctly", () => {
      const file = readFileSync(findLanceDataFile("simple_float64.lance"));
      const fileBytes = file.buffer.slice(file.byteOffset, file.byteOffset + file.length);
      const footer = parseFooter(fileBytes)!;

      const metaStart = Number(footer.columnMetaStart);
      const metaEnd = file.length - FOOTER_SIZE;
      const columns = parseColumnMetaFromProtobuf(fileBytes.slice(metaStart, metaEnd), footer.numColumns);

      expect(columns.length).toBe(1);
      const col = columns[0];

      const page = col.pages[0];
      if (page && page.byteLength > 0) {
        const pageData = fileBytes.slice(Number(page.byteOffset), Number(page.byteOffset) + page.byteLength);
        // Decode as float64 (expected: [1.5, 2.5, 3.5, 4.5, 5.5])
        const values = decodePage(new Uint8Array(pageData).buffer, "float64", page.nullCount, page.rowCount);
        expect(values.length).toBe(5);
        const expected = [1.5, 2.5, 3.5, 4.5, 5.5];
        for (let i = 0; i < expected.length; i++) {
          expect(values[i]).toBeCloseTo(expected[i], 10);
        }
      }
    });
  });
});

// ============================================================================
// 5. Filter result correctness
// ============================================================================

describe("Filter result correctness", () => {
  describe("matchesFilter returns correct results for all operators", () => {
    it("eq: exact match only", () => {
      expect(matchesFilter(42, { column: "x", op: "eq", value: 42 })).toBe(true);
      expect(matchesFilter(41, { column: "x", op: "eq", value: 42 })).toBe(false);
      expect(matchesFilter("hello", { column: "x", op: "eq", value: "hello" })).toBe(true);
      expect(matchesFilter("world", { column: "x", op: "eq", value: "hello" })).toBe(false);
    });

    it("neq: excludes exact match", () => {
      expect(matchesFilter(42, { column: "x", op: "neq", value: 42 })).toBe(false);
      expect(matchesFilter(43, { column: "x", op: "neq", value: 42 })).toBe(true);
    });

    it("gt/gte: strict and inclusive", () => {
      expect(matchesFilter(10, { column: "x", op: "gt", value: 10 })).toBe(false);
      expect(matchesFilter(11, { column: "x", op: "gt", value: 10 })).toBe(true);
      expect(matchesFilter(10, { column: "x", op: "gte", value: 10 })).toBe(true);
      expect(matchesFilter(9, { column: "x", op: "gte", value: 10 })).toBe(false);
    });

    it("lt/lte: strict and inclusive", () => {
      expect(matchesFilter(10, { column: "x", op: "lt", value: 10 })).toBe(false);
      expect(matchesFilter(9, { column: "x", op: "lt", value: 10 })).toBe(true);
      expect(matchesFilter(10, { column: "x", op: "lte", value: 10 })).toBe(true);
      expect(matchesFilter(11, { column: "x", op: "lte", value: 10 })).toBe(false);
    });

    it("in: membership check", () => {
      expect(matchesFilter(2, { column: "x", op: "in", value: [1, 2, 3] })).toBe(true);
      expect(matchesFilter(4, { column: "x", op: "in", value: [1, 2, 3] })).toBe(false);
      expect(matchesFilter("b", { column: "x", op: "in", value: ["a", "b", "c"] })).toBe(true);
    });

    it("null values never match any filter", () => {
      expect(matchesFilter(null, { column: "x", op: "eq", value: 0 })).toBe(false);
      expect(matchesFilter(null, { column: "x", op: "gt", value: -999 })).toBe(false);
      expect(matchesFilter(null, { column: "x", op: "in", value: [null as any] })).toBe(false);
    });
  });

  describe("filtered decoding from real Parquet data", () => {
    it("filtering INT64 column returns exactly matching rows", () => {
      const filePath = join(GENERATED, "bench_100k_3col.parquet");
      const file = readFileSync(filePath);
      const { footerBuf, fileSize } = loadParquetFooterBytes(filePath);
      const meta = parseParquetFooter(footerBuf)!;
      const tableMeta = parquetMetaToTableMeta(meta, "bench_100k_3col.parquet", BigInt(fileSize));

      const idCol = tableMeta.columns.find(c => c.name === "id")!;
      // Decode only the last row group (IDs 90000-99999)
      const lastPage = idCol.pages[idCol.pages.length - 1];
      const offset = Number(lastPage.byteOffset);
      const chunkBuf = file.buffer.slice(file.byteOffset + offset, file.byteOffset + offset + lastPage.byteLength);
      const allValues = decodeParquetColumnChunk(chunkBuf, lastPage.encoding!, idCol.dtype, lastPage.rowCount, mockWasm);

      // Apply filter: id > 99990n
      const threshold = 99990n;
      const filtered = allValues.filter(v => v !== null && (v as bigint) > threshold);

      // IDs 90000..99999 → those > 99990 are 99991..99999 = 9 rows
      expect(filtered.length).toBe(9);
      for (const v of filtered) {
        expect(v as bigint > threshold).toBe(true);
      }
    });

    it("combined page skip + in-memory filter yields consistent results", () => {
      const filePath = join(GENERATED, "bench_100k_numeric.parquet");
      const file = readFileSync(filePath);
      const { footerBuf, fileSize } = loadParquetFooterBytes(filePath);
      const meta = parseParquetFooter(footerBuf)!;
      const tableMeta = parquetMetaToTableMeta(meta, "bench_100k_numeric.parquet", BigInt(fileSize));

      const idCol = tableMeta.columns.find(c => c.name === "id")!;
      const filter = { column: "id", op: "gt" as const, value: 99000 };

      let totalMatchingRows = 0;
      for (const page of idCol.pages) {
        // First try page skip
        if (canSkipPage(page, [filter], "id")) continue;

        // Decode and filter in-memory
        const offset = Number(page.byteOffset);
        const chunkBuf = file.buffer.slice(file.byteOffset + offset, file.byteOffset + offset + page.byteLength);
        const values = decodeParquetColumnChunk(chunkBuf, page.encoding!, idCol.dtype, page.rowCount, mockWasm);
        const matching = values.filter(v => v !== null && Number(v) > 99000);
        totalMatchingRows += matching.length;
      }

      // IDs 0..99999, those > 99000 are 99001..99999 = 999 rows
      expect(totalMatchingRows).toBe(999);
    });
  });
});

// ============================================================================
// 6. Cross-format consistency
// ============================================================================

describe("Cross-format consistency", () => {
  it("Lance and Parquet agree on column types for numeric data", () => {
    // Parquet bench_100k_numeric id column uses int64
    const { footerBuf, fileSize } = loadParquetFooterBytes(join(GENERATED, "bench_100k_numeric.parquet"));
    const parquetMeta = parseParquetFooter(footerBuf)!;
    const parquetTable = parquetMetaToTableMeta(parquetMeta, "bench_100k_numeric.parquet", BigInt(fileSize));

    const parquetIdCol = parquetTable.columns.find(c => c.name === "id")!;
    expect(parquetIdCol.dtype).toBe("int64");

    // Lance simple_int64 manifest says the column is int64 too
    const manifestPath = join(FIXTURES, "simple_int64.lance", "_versions", "1.manifest");
    const manifestFile = readFileSync(manifestPath);
    const manifest = parseManifest(manifestFile.buffer.slice(manifestFile.byteOffset, manifestFile.byteOffset + manifestFile.length))!;
    const idField = manifest.schema.find(f => f.name === "id")!;
    expect(idField).toBeDefined();
    // logicalType should indicate int64
    expect(idField.logicalType).toMatch(/int64/i);
  });

  it("Lance mixed_types column metadata matches expected.json", () => {
    const expected = JSON.parse(readFileSync(join(FIXTURES, "mixed_types.expected.json"), "utf8"));

    // Verify from manifest
    const manifestPath = join(FIXTURES, "mixed_types.lance", "_versions", "1.manifest");
    const manifestFile = readFileSync(manifestPath);
    const manifest = parseManifest(manifestFile.buffer.slice(manifestFile.byteOffset, manifestFile.byteOffset + manifestFile.length))!;

    expect(manifest.totalRows).toBe(expected.row_count);
    const manifestColNames = manifest.schema.map(f => f.name);
    for (const col of expected.columns) {
      expect(manifestColNames).toContain(col);
    }
  });

  it("Lance with_nulls data file has correct number of columns", () => {
    const expected = JSON.parse(readFileSync(join(FIXTURES, "with_nulls.expected.json"), "utf8"));

    const dataDir = join(FIXTURES, "with_nulls.lance", "data");
    const files = readdirSync(dataDir);
    const lanceFile = files.find(f => f.endsWith(".lance"))!;
    const file = readFileSync(join(dataDir, lanceFile));
    const fileBytes = file.buffer.slice(file.byteOffset, file.byteOffset + file.length);
    const footer = parseFooter(fileBytes)!;
    const metaStart = Number(footer.columnMetaStart);
    const metaEnd = file.length - FOOTER_SIZE;
    const columns = parseColumnMetaFromProtobuf(fileBytes.slice(metaStart, metaEnd), footer.numColumns);

    // Number of columns in data file matches expected
    expect(columns.length).toBe(expected.columns.length);

    // Verify the manifest reports the correct row count and column names
    const manifestPath = join(FIXTURES, "with_nulls.lance", "_versions", "1.manifest");
    const manifestFile = readFileSync(manifestPath);
    const manifest = parseManifest(manifestFile.buffer.slice(manifestFile.byteOffset, manifestFile.byteOffset + manifestFile.length))!;
    expect(manifest.totalRows).toBe(expected.row_count);
    const fieldNames = manifest.schema.map(f => f.name);
    expect(fieldNames).toContain("id");
    expect(fieldNames).toContain("value");
  });

  it("decodePlainValues produces correct types for each DataType", () => {
    // INT32
    const int32Buf = new Uint8Array(8);
    new DataView(int32Buf.buffer).setInt32(0, 42, true);
    new DataView(int32Buf.buffer).setInt32(4, -1, true);
    expect(decodePlainValues(int32Buf, "int32", 2)).toEqual([42, -1]);

    // INT64
    const int64Buf = new Uint8Array(8);
    new DataView(int64Buf.buffer).setBigInt64(0, 123456789n, true);
    expect(decodePlainValues(int64Buf, "int64", 1)).toEqual([123456789n]);

    // FLOAT64
    const f64Buf = new Uint8Array(8);
    new DataView(f64Buf.buffer).setFloat64(0, 3.14159, true);
    const f64Result = decodePlainValues(f64Buf, "float64", 1);
    expect(f64Result.length).toBe(1);
    expect(f64Result[0]).toBeCloseTo(3.14159, 10);

    // BOOL
    const boolBuf = new Uint8Array([0b00000101]); // bits 0 and 2 set
    expect(decodePlainValues(boolBuf, "bool", 3)).toEqual([true, false, true]);

    // UTF8
    const encoder = new TextEncoder();
    const strBytes = encoder.encode("test");
    const utf8Buf = new Uint8Array(4 + strBytes.length);
    new DataView(utf8Buf.buffer).setUint32(0, strBytes.length, true);
    utf8Buf.set(strBytes, 4);
    expect(decodePlainValues(utf8Buf, "utf8", 1)).toEqual(["test"]);
  });
});
