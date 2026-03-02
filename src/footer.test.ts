import { describe, it, expect } from "vitest";
import { parseFooter, parseColumnMetaFromProtobuf, readVarint, LANCE_MAGIC, FOOTER_SIZE } from "./footer.js";

describe("parseFooter", () => {
  /** Build a valid 40-byte Lance footer */
  function buildFooter(opts: {
    columnMetaStart?: bigint;
    columnMetaOffsetsStart?: bigint;
    globalBuffOffsetsStart?: bigint;
    numGlobalBuffers?: number;
    numColumns?: number;
    majorVersion?: number;
    minorVersion?: number;
  } = {}): ArrayBuffer {
    const buf = new ArrayBuffer(FOOTER_SIZE);
    const view = new DataView(buf);

    view.setBigUint64(0, opts.columnMetaStart ?? 100n, true);
    view.setBigUint64(8, opts.columnMetaOffsetsStart ?? 200n, true);
    view.setBigUint64(16, opts.globalBuffOffsetsStart ?? 300n, true);
    view.setUint32(24, opts.numGlobalBuffers ?? 0, true);
    view.setUint32(28, opts.numColumns ?? 3, true);
    view.setUint16(32, opts.majorVersion ?? 2, true);
    view.setUint16(34, opts.minorVersion ?? 0, true);
    view.setUint32(36, LANCE_MAGIC, true); // "LANC"

    return buf;
  }

  it("parses a valid 40-byte footer", () => {
    const buf = buildFooter({ numColumns: 5, majorVersion: 2, minorVersion: 1 });
    const footer = parseFooter(buf);

    expect(footer).not.toBeNull();
    expect(footer!.numColumns).toBe(5);
    expect(footer!.majorVersion).toBe(2);
    expect(footer!.minorVersion).toBe(1);
    expect(footer!.columnMetaStart).toBe(100n);
    expect(footer!.columnMetaOffsetsStart).toBe(200n);
  });

  it("returns null for wrong magic number", () => {
    const buf = buildFooter();
    const view = new DataView(buf);
    view.setUint32(36, 0xdeadbeef, true);
    expect(parseFooter(buf)).toBeNull();
  });

  it("returns null for too-small buffer", () => {
    expect(parseFooter(new ArrayBuffer(10))).toBeNull();
  });

  it("handles footer embedded in larger buffer", () => {
    const prefix = new Uint8Array(100);
    const footer = new Uint8Array(buildFooter({ numColumns: 7 }));
    const combined = new Uint8Array(prefix.length + footer.length);
    combined.set(prefix);
    combined.set(footer, prefix.length);

    const parsed = parseFooter(combined.buffer);
    expect(parsed).not.toBeNull();
    expect(parsed!.numColumns).toBe(7);
  });
});

describe("readVarint", () => {
  it("reads single-byte varint", () => {
    const bytes = new Uint8Array([42]);
    const { value, bytesRead } = readVarint(bytes, 0);
    expect(value).toBe(42);
    expect(bytesRead).toBe(1);
  });

  it("reads multi-byte varint", () => {
    // 300 = 0b100101100 → varint: [0xAC, 0x02]
    const bytes = new Uint8Array([0xac, 0x02]);
    const { value, bytesRead } = readVarint(bytes, 0);
    expect(value).toBe(300);
    expect(bytesRead).toBe(2);
  });

  it("reads varint at offset", () => {
    const bytes = new Uint8Array([0xff, 0xff, 42]);
    const { value, bytesRead } = readVarint(bytes, 2);
    expect(value).toBe(42);
    expect(bytesRead).toBe(1);
  });
});

describe("parseColumnMetaFromProtobuf", () => {
  /** Build a minimal protobuf column metadata message */
  function buildColumnMeta(name: string, dtypeEnum: number): Uint8Array {
    // Protobuf message: field 1 (string) = name, field 2 (varint) = dtype
    const nameBytes = new TextEncoder().encode(name);
    const parts: number[] = [];

    // Field 1: string (tag = 0x0a = field 1, wire type 2)
    parts.push(0x0a, nameBytes.length, ...nameBytes);

    // Field 2: varint (tag = 0x10 = field 2, wire type 0)
    parts.push(0x10, dtypeEnum);

    return new Uint8Array(parts);
  }

  it("parses a single column", () => {
    const colMeta = buildColumnMeta("age", 2); // int32
    // Wrap in length-prefixed message
    const msg = new Uint8Array([colMeta.length, ...colMeta]);

    const columns = parseColumnMetaFromProtobuf(msg.buffer, 1);
    expect(columns).toHaveLength(1);
    expect(columns[0].name).toBe("age");
    expect(columns[0].dtype).toBe("int32");
  });

  it("parses multiple columns", () => {
    const col1 = buildColumnMeta("id", 3);   // int64
    const col2 = buildColumnMeta("score", 9); // float32
    const msg = new Uint8Array([
      col1.length, ...col1,
      col2.length, ...col2,
    ]);

    const columns = parseColumnMetaFromProtobuf(msg.buffer, 2);
    expect(columns).toHaveLength(2);
    expect(columns[0].name).toBe("id");
    expect(columns[0].dtype).toBe("int64");
    expect(columns[1].name).toBe("score");
    expect(columns[1].dtype).toBe("float32");
  });

  it("extracts listDimension from field 4", () => {
    const nameBytes = new TextEncoder().encode("embedding");
    const parts: number[] = [];
    parts.push(0x0a, nameBytes.length, ...nameBytes); // field 1: name
    parts.push(0x10, 14); // field 2: fixed_size_list
    parts.push(0x20, 0x80, 0x01); // field 4 (tag 0x20 = field 4, wire type 0): dimension = 128 (varint)

    const colMeta = new Uint8Array(parts);
    const msg = new Uint8Array([colMeta.length, ...colMeta]);

    const columns = parseColumnMetaFromProtobuf(msg.buffer, 1);
    expect(columns).toHaveLength(1);
    expect(columns[0].name).toBe("embedding");
    expect(columns[0].dtype).toBe("fixed_size_list");
    expect(columns[0].listDimension).toBe(128);
  });
});
