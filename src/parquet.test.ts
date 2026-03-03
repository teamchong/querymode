import { describe, it, expect } from "vitest";
import { detectFormat, getParquetFooterLength, parseParquetFooter, parquetMetaToTableMeta, PARQUET_MAGIC, ThriftReader } from "./parquet.js";
import { LANCE_MAGIC } from "./footer.js";

describe("detectFormat", () => {
  it("detects PAR1 magic", () => {
    const buf = new ArrayBuffer(8);
    const view = new DataView(buf);
    view.setUint32(4, PARQUET_MAGIC, true);
    expect(detectFormat(buf)).toBe("parquet");
  });

  it("detects LANC magic", () => {
    const buf = new ArrayBuffer(8);
    const view = new DataView(buf);
    view.setUint32(4, LANCE_MAGIC, true);
    expect(detectFormat(buf)).toBe("lance");
  });

  it("returns null for unknown magic", () => {
    const buf = new ArrayBuffer(8);
    expect(detectFormat(buf)).toBe(null);
  });

  it("returns null for buffer too small", () => {
    const buf = new ArrayBuffer(2);
    expect(detectFormat(buf)).toBe(null);
  });
});

describe("getParquetFooterLength", () => {
  it("reads footer length from last 8 bytes", () => {
    const buf = new ArrayBuffer(8);
    const view = new DataView(buf);
    view.setUint32(0, 1234, true);
    view.setUint32(4, PARQUET_MAGIC, true);
    expect(getParquetFooterLength(buf)).toBe(1234);
  });

  it("reads from larger buffer", () => {
    const buf = new ArrayBuffer(40);
    const view = new DataView(buf);
    view.setUint32(32, 5678, true);
    view.setUint32(36, PARQUET_MAGIC, true);
    expect(getParquetFooterLength(buf)).toBe(5678);
  });

  it("returns null for non-PAR1 magic", () => {
    const buf = new ArrayBuffer(8);
    expect(getParquetFooterLength(buf)).toBe(null);
  });

  it("returns null for too-small buffer", () => {
    const buf = new ArrayBuffer(4);
    expect(getParquetFooterLength(buf)).toBe(null);
  });
});

describe("parseParquetFooter", () => {
  it("returns null for empty buffer", () => {
    expect(parseParquetFooter(new ArrayBuffer(0))).toBe(null);
  });

  it("parses a minimal footer with version field", () => {
    // Thrift compact: field 1, type I32 (5), delta=1 → byte = (1<<4)|5 = 0x15
    // zigzag(1) = 0x02, then struct end 0x00
    const bytes = new Uint8Array([0x15, 0x02, 0x00]);
    const result = parseParquetFooter(bytes.buffer);
    expect(result).not.toBe(null);
    expect(result!.version).toBe(1);
    expect(result!.schema).toEqual([]);
    expect(result!.rowGroups).toEqual([]);
  });
});

describe("parquetMetaToTableMeta", () => {
  it("converts simple metadata to TableMeta", () => {
    const meta = {
      version: 2,
      schema: [
        { name: "root", numChildren: 2 },
        { name: "id", type: 1 },
        { name: "name", type: 6, convertedType: 0 },
      ],
      numRows: 100,
      rowGroups: [{
        columns: [
          {
            type: 1, codec: 0, numValues: 100,
            totalCompressedSize: 400, totalUncompressedSize: 400,
            dataPageOffset: 0n, pathInSchema: ["id"],
          },
          {
            type: 6, codec: 1, numValues: 100,
            totalCompressedSize: 800, totalUncompressedSize: 1200,
            dataPageOffset: 400n, dictionaryPageOffset: 200n,
            pathInSchema: ["name"],
          },
        ],
        numRows: 100,
      }],
    };

    const tableMeta = parquetMetaToTableMeta(meta, "data/test.parquet", 2048n);
    expect(tableMeta.format).toBe("parquet");
    expect(tableMeta.totalRows).toBe(100);
    expect(tableMeta.columns).toHaveLength(2);

    const idCol = tableMeta.columns.find(c => c.name === "id")!;
    expect(idCol.dtype).toBe("int32");
    expect(idCol.pages).toHaveLength(1);
    expect(idCol.pages[0].encoding?.encoding).toBe("PLAIN");
    expect(idCol.pages[0].encoding?.compression).toBe("UNCOMPRESSED");

    const nameCol = tableMeta.columns.find(c => c.name === "name")!;
    expect(nameCol.dtype).toBe("utf8");
    expect(nameCol.pages[0].encoding?.encoding).toBe("RLE_DICTIONARY");
    expect(nameCol.pages[0].encoding?.compression).toBe("SNAPPY");
    expect(nameCol.pages[0].encoding?.dictionaryPageOffset).toBe(200n);
  });

  it("handles multiple row groups", () => {
    const meta = {
      version: 2,
      schema: [
        { name: "schema", numChildren: 1 },
        { name: "age", type: 1 },
      ],
      numRows: 200,
      rowGroups: [
        {
          columns: [{ type: 1, codec: 0, numValues: 100, totalCompressedSize: 400, totalUncompressedSize: 400, dataPageOffset: 0n, pathInSchema: ["age"] }],
          numRows: 100,
        },
        {
          columns: [{ type: 1, codec: 0, numValues: 100, totalCompressedSize: 400, totalUncompressedSize: 400, dataPageOffset: 400n, pathInSchema: ["age"] }],
          numRows: 100,
        },
      ],
    };

    const tableMeta = parquetMetaToTableMeta(meta, "test.parquet", 1024n);
    expect(tableMeta.columns).toHaveLength(1);
    expect(tableMeta.columns[0].pages).toHaveLength(2);
    expect(tableMeta.totalRows).toBe(200);
  });
});

describe("ThriftReader", () => {
  it("reads varint values", () => {
    // varint encoding of 300: 0xAC 0x02
    const r = new ThriftReader(new Uint8Array([0xAC, 0x02]));
    expect(r.readVarint()).toBe(300);
  });

  it("reads zigzag i32 values", () => {
    // Thrift compact: field 1 i32, delta=1 → (1<<4)|5 = 0x15
    // zigzag(-1) = 1 → varint(1) = 0x01
    const r = new ThriftReader(new Uint8Array([0x15, 0x01, 0x00]));
    const f = r.nextField();
    expect(f).not.toBe(null);
    expect(f!.fieldId).toBe(1);
    expect(r.readI32()).toBe(-1);
  });

  it("reads list headers", () => {
    // List header: high nibble = size (3), low nibble = elem type (5=i32)
    const r = new ThriftReader(new Uint8Array([(3 << 4) | 5]));
    const { size, elemType } = r.readListHeader();
    expect(size).toBe(3);
    expect(elemType).toBe(5);
  });

  it("reads large list headers with varint size", () => {
    // size=0xF means "read varint next", elem type=8 (binary)
    const r = new ThriftReader(new Uint8Array([0xF8, 20])); // size=0xF, elemType=8, then varint(20)
    const { size, elemType } = r.readListHeader();
    expect(size).toBe(20);
    expect(elemType).toBe(8);
  });

  it("readStruct saves/restores lastFieldId", () => {
    const r = new ThriftReader(new Uint8Array([0x00])); // just struct end
    r.lastFieldId = 42;
    r.readStruct(inner => {
      expect(inner.lastFieldId).toBe(0);
      return null;
    });
    expect(r.lastFieldId).toBe(42);
  });

  it("skips unknown fields", () => {
    // Field 1: i32 (type 5), value zigzag(0)=0x00
    // Field 2: binary (type 8), length=2, bytes=[0xAB, 0xCD]
    // Struct end: 0x00
    const r = new ThriftReader(new Uint8Array([
      0x15, 0x00,           // field 1, i32, value=0
      0x18, 0x02, 0xAB, 0xCD, // field 2, binary, len=2
      0x00,                 // struct end
    ]));
    const f1 = r.nextField()!;
    expect(f1.fieldId).toBe(1);
    r.skip(f1.typeId); // skip i32
    const f2 = r.nextField()!;
    expect(f2.fieldId).toBe(2);
    r.skip(f2.typeId); // skip binary
    expect(r.nextField()).toBe(null); // struct end
  });
});
