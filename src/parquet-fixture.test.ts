import { readFileSync, existsSync } from "node:fs";
import { describe, it, expect } from "vitest";
import { parseParquetFooter, parquetMetaToTableMeta } from "./parquet.js";

const FIXTURES = "wasm/tests/fixtures";

function loadParquetMeta(path: string) {
  const buf = readFileSync(path);
  const footerLen = new DataView(buf.buffer, buf.byteOffset).getUint32(buf.length - 8, true);
  const footerBytes = new Uint8Array(buf.buffer, buf.byteOffset + buf.length - 8 - footerLen, footerLen);
  const meta = parseParquetFooter(footerBytes.buffer.slice(footerBytes.byteOffset, footerBytes.byteOffset + footerBytes.length))!;
  return { meta, fileSize: BigInt(buf.length) };
}

describe.skipIf(!existsSync(`${FIXTURES}/simple_snappy.parquet`))("parquet fixture parsing", () => {
  it("plain encoding parses correctly", () => {
    const { meta, fileSize } = loadParquetMeta(`${FIXTURES}/simple_plain.parquet`);
    const tableMeta = parquetMetaToTableMeta(meta, "simple_plain.parquet", fileSize);
    expect(tableMeta.totalRows).toBe(5);
    expect(tableMeta.columns).toHaveLength(3);
    expect(tableMeta.columns.map(c => c.name)).toEqual(["id", "name", "value"]);
    expect(tableMeta.columns.map(c => c.dtype)).toEqual(["int64", "utf8", "float64"]);
    for (const c of tableMeta.columns) {
      expect(c.pages).toHaveLength(1);
      expect(c.pages[0].rowCount).toBe(5);
    }
  });

  it("snappy compression parses correctly", () => {
    const { meta, fileSize } = loadParquetMeta(`${FIXTURES}/simple_snappy.parquet`);
    const tableMeta = parquetMetaToTableMeta(meta, "simple_snappy.parquet", fileSize);
    expect(tableMeta.totalRows).toBe(5);
    expect(tableMeta.columns).toHaveLength(3);
    expect(tableMeta.columns.map(c => c.name)).toEqual(["id", "name", "value"]);
    expect(tableMeta.columns.map(c => c.dtype)).toEqual(["int64", "utf8", "float64"]);
    for (const c of tableMeta.columns) {
      expect(c.pages).toHaveLength(1);
      expect(c.pages[0].rowCount).toBe(5);
      expect(c.pages[0].encoding?.compression).toBe("SNAPPY");
    }
  });

  it("default encoding parses correctly", () => {
    if (!existsSync(`${FIXTURES}/simple.parquet`)) return;
    const { meta, fileSize } = loadParquetMeta(`${FIXTURES}/simple.parquet`);
    const tableMeta = parquetMetaToTableMeta(meta, "simple.parquet", fileSize);
    expect(tableMeta.totalRows).toBe(5);
    expect(tableMeta.columns).toHaveLength(3);
    for (const c of tableMeta.columns) {
      expect(c.pages).toHaveLength(1);
      expect(c.pages[0].rowCount).toBe(5);
    }
  });

  it("100k benchmark parquet parses correctly", () => {
    if (!existsSync(`${FIXTURES}/benchmark_100k.parquet`)) return;
    const { meta, fileSize } = loadParquetMeta(`${FIXTURES}/benchmark_100k.parquet`);
    const tableMeta = parquetMetaToTableMeta(meta, "benchmark_100k.parquet", fileSize);
    expect(tableMeta.totalRows).toBe(100_000);
    expect(tableMeta.columns.length).toBeGreaterThan(0);
    for (const c of tableMeta.columns) {
      expect(c.pages.length).toBeGreaterThan(0);
    }
  });
});
