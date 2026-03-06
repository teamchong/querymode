import { describe, it, expect } from "vitest";
import { formatResultSummary, formatExplain, formatBytes } from "./format.js";
import type { QueryResult, ExplainResult } from "./types.js";

describe("formatBytes", () => {
  it("formats 0 bytes", () => {
    expect(formatBytes(0)).toBe("0B");
  });

  it("formats bytes", () => {
    expect(formatBytes(500)).toBe("500B");
  });

  it("formats kilobytes", () => {
    expect(formatBytes(1024)).toBe("1.0KB");
    expect(formatBytes(5120)).toBe("5.0KB");
    expect(formatBytes(12345)).toBe("12KB");
  });

  it("formats megabytes", () => {
    expect(formatBytes(1024 * 1024)).toBe("1.0MB");
    expect(formatBytes(1.5 * 1024 * 1024)).toBe("1.5MB");
  });

  it("formats gigabytes", () => {
    expect(formatBytes(1024 * 1024 * 1024)).toBe("1.0GB");
    expect(formatBytes(2.5 * 1024 * 1024 * 1024)).toBe("2.5GB");
  });
});

describe("formatResultSummary", () => {
  it("formats basic result", () => {
    const result: QueryResult = {
      rows: Array(20).fill({}),
      rowCount: 20,
      columns: ["a"],
      bytesRead: 12288,
      pagesSkipped: 847,
      durationMs: 3.2,
    };
    const summary = formatResultSummary(result);
    expect(summary).toContain("20 rows");
    expect(summary).toContain("3.2ms");
    expect(summary).toContain("847 pages skipped");
    expect(summary).toContain("12KB read");
  });

  it("handles cache hit", () => {
    const result: QueryResult = {
      rows: [{}],
      rowCount: 1,
      columns: ["a"],
      bytesRead: 0,
      pagesSkipped: 0,
      durationMs: 0.1,
      cacheHit: true,
    };
    expect(formatResultSummary(result)).toContain("cache hit");
  });

  it("includes optional timing fields", () => {
    const result = {
      rows: [{}],
      rowCount: 1,
      columns: ["a"],
      bytesRead: 100,
      pagesSkipped: 0,
      durationMs: 5.0,
      r2ReadMs: 2.5,
      wasmExecMs: 1.2,
      spillBytesWritten: 4096,
    };
    const summary = formatResultSummary(result);
    expect(summary).toContain("r2: 2.5ms");
    expect(summary).toContain("wasm: 1.2ms");
    expect(summary).toContain("4.0KB spilled");
  });

  it("includes local timing info", () => {
    const result = {
      rows: [{}],
      rowCount: 1,
      columns: ["a"],
      bytesRead: 0,
      pagesSkipped: 0,
      durationMs: 10.0,
      scanMs: 3.0,
      pipelineMs: 5.0,
      metaMs: 2.0,
    };
    const summary = formatResultSummary(result);
    expect(summary).toContain("scan: 3.0ms");
    expect(summary).toContain("pipeline: 5.0ms");
    expect(summary).toContain("meta: 2.0ms");
  });

  it("singular row", () => {
    const result: QueryResult = {
      rows: [{}],
      rowCount: 1,
      columns: ["a"],
      bytesRead: 0,
      pagesSkipped: 0,
      durationMs: 1.0,
    };
    expect(formatResultSummary(result)).toContain("1 row in");
  });
});

describe("formatExplain", () => {
  it("formats a complete explain plan", () => {
    const plan: ExplainResult = {
      table: "orders.parquet",
      format: "parquet",
      totalRows: 5000000,
      columns: [
        { name: "name", dtype: "utf8", pages: 12, bytes: 49152 },
        { name: "amount", dtype: "int64", pages: 8, bytes: 32768 },
      ],
      pagesTotal: 32,
      pagesSkipped: 20,
      pagesScanned: 12,
      estimatedBytes: 131072,
      estimatedR2Reads: 4,
      estimatedRows: 2000000,
      fragments: 1,
      filters: [
        { column: "amount", op: "gt", pushable: true },
        { column: "region", op: "in", pushable: false },
      ],
      metaCached: true,
    };

    const output = formatExplain(plan);
    expect(output).toContain("Table: orders.parquet (parquet)");
    expect(output).toContain("5,000,000");
    expect(output).toContain("Fragments: 1");
    expect(output).toContain("Columns: 2 scanned");
    expect(output).toContain("name");
    expect(output).toContain("utf8");
    expect(output).toContain("amount gt  [pushable]");
    expect(output).toContain("region in  [not pushable]");
    expect(output).toContain("20 skipped (62.5%)");
    expect(output).toContain("128KB across 4 reads");
    expect(output).toContain("Meta: cached");
  });

  it("handles no filters", () => {
    const plan: ExplainResult = {
      table: "test.lance",
      format: "lance",
      totalRows: 100,
      columns: [{ name: "id", dtype: "int64", pages: 1, bytes: 800 }],
      pagesTotal: 1,
      pagesSkipped: 0,
      pagesScanned: 1,
      estimatedBytes: 800,
      estimatedR2Reads: 1,
      estimatedRows: 100,
      fragments: 1,
      filters: [],
      metaCached: false,
    };

    const output = formatExplain(plan);
    expect(output).not.toContain("Filters:");
    expect(output).not.toContain("Meta: cached");
  });
});
