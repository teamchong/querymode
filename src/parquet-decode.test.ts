import { describe, it, expect } from "vitest";
import { decompressSnappy, decodeParquetColumnChunk, decodePlainValues } from "./parquet-decode.js";

describe("decompressSnappy", () => {
  it("decompresses a literal-only block", () => {
    // varint(5) = uncompressed length
    // literal tag: (4 << 2) | 0 = 0x10 → length = (0x10>>2)+1 = 5
    const input = new Uint8Array([
      5,           // varint: uncompressed length = 5
      0x10,        // literal tag: length = (0x10 >> 2) + 1 = 5
      72, 101, 108, 108, 111, // "Hello"
    ]);
    const output = decompressSnappy(input);
    expect(output.length).toBe(5);
    expect(new TextDecoder().decode(output)).toBe("Hello");
  });

  it("handles zero-length input", () => {
    const input = new Uint8Array([0]); // uncompressed length = 0
    const output = decompressSnappy(input);
    expect(output.length).toBe(0);
  });

  it("decompresses multiple literals", () => {
    // Two separate literal blocks: "AB" then "CD" → "ABCD"
    const input = new Uint8Array([
      4,           // uncompressed length = 4
      0x04,        // literal tag: length = (0x04>>2)+1 = 2
      65, 66,      // "AB"
      0x04,        // literal tag: length = 2
      67, 68,      // "CD"
    ]);
    const output = decompressSnappy(input);
    expect(output.length).toBe(4);
    expect(new TextDecoder().decode(output)).toBe("ABCD");
  });

  it("decompresses a block with copy-2 offset", () => {
    // "ABAB" = literal "AB" + copy-2 (len=2, offset=2)
    // copy-2 tag: ((len-1)<<2)|2 = ((2-1)<<2)|2 = 0x06
    // offset = 2 as 2 bytes LE = [0x02, 0x00]
    const input = new Uint8Array([
      4,           // uncompressed length = 4
      0x04,        // literal: 2 bytes
      65, 66,      // "AB"
      0x06,        // copy-2: len=((0x06>>2)&63)+1=2, offset in next 2 bytes
      0x02, 0x00,  // offset = 2
    ]);
    const output = decompressSnappy(input);
    expect(output.length).toBe(4);
    expect(new TextDecoder().decode(output)).toBe("ABAB");
  });
});

describe("decodeParquetColumnChunk", () => {
  it("returns empty array for empty buffer", () => {
    const result = decodeParquetColumnChunk(
      new ArrayBuffer(0),
      { encoding: "PLAIN", compression: "UNCOMPRESSED" },
      "int32",
      0,
    );
    expect(result).toEqual([]);
  });

  it("returns empty array for zero numValues", () => {
    const buf = new ArrayBuffer(10);
    const result = decodeParquetColumnChunk(
      buf,
      { encoding: "PLAIN", compression: "UNCOMPRESSED" },
      "int32",
      0,
    );
    expect(result).toEqual([]);
  });
});

describe("decodePlainValues", () => {
  it("decodes int32 values", () => {
    const buf = new ArrayBuffer(12);
    const view = new DataView(buf);
    view.setInt32(0, 100, true);
    view.setInt32(4, -50, true);
    view.setInt32(8, 0, true);
    const result = decodePlainValues(new Uint8Array(buf), "int32", 3);
    expect(result).toEqual([100, -50, 0]);
  });

  it("decodes float64 values", () => {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setFloat64(0, 3.14, true);
    view.setFloat64(8, -2.718, true);
    const result = decodePlainValues(new Uint8Array(buf), "float64", 2);
    expect(result).toHaveLength(2);
    expect(result[0]).toBeCloseTo(3.14);
    expect(result[1]).toBeCloseTo(-2.718);
  });

  it("decodes bool values from bit-packed bytes", () => {
    // 0b00000101 = bits 0,2 set → true, false, true, false, false, false, false, false
    const result = decodePlainValues(new Uint8Array([0x05]), "bool", 3);
    expect(result).toEqual([true, false, true]);
  });

  it("decodes utf8 strings", () => {
    const enc = new TextEncoder();
    const s1 = enc.encode("hi");
    const s2 = enc.encode("bye");
    const buf = new ArrayBuffer(4 + s1.length + 4 + s2.length);
    const view = new DataView(buf);
    let pos = 0;
    view.setUint32(pos, s1.length, true); pos += 4;
    new Uint8Array(buf, pos).set(s1); pos += s1.length;
    view.setUint32(pos, s2.length, true); pos += 4;
    new Uint8Array(buf, pos).set(s2);
    const result = decodePlainValues(new Uint8Array(buf), "utf8", 2);
    expect(result).toEqual(["hi", "bye"]);
  });

  it("respects numValues cap", () => {
    const buf = new ArrayBuffer(12);
    const view = new DataView(buf);
    view.setInt32(0, 1, true);
    view.setInt32(4, 2, true);
    view.setInt32(8, 3, true);
    const result = decodePlainValues(new Uint8Array(buf), "int32", 2);
    expect(result).toEqual([1, 2]);
  });

  it("decodes int8 values", () => {
    const result = decodePlainValues(new Uint8Array([0x01, 0xFF, 0x80]), "int8", 3);
    expect(result).toEqual([1, -1, -128]);
  });

  it("decodes binary as hex strings", () => {
    const buf = new ArrayBuffer(7);
    const view = new DataView(buf);
    view.setUint32(0, 3, true); // length = 3
    new Uint8Array(buf, 4).set([0xAB, 0xCD, 0xEF]);
    const result = decodePlainValues(new Uint8Array(buf), "binary", 1);
    expect(result).toEqual(["abcdef"]);
  });
});

describe("decompressSnappy - copy-1 offset", () => {
  it("decompresses a block with copy-1 offset", () => {
    // "ABCABC" = literal "ABC" + copy-1 (len=4+3=7? no, len=((tag>>2)&7)+4)
    // Actually copy-1: length = ((tag >> 2) & 0x07) + 4, offset = ((tag & 0xe0) << 3) | next_byte
    // We want len=3, but min copy-1 length is 4. Let's do "ABCDABCD"
    // literal "ABCD" + copy-1 len=4 offset=4
    // tag: tagType=1, length_minus_4=0, offset_high_bits=0 → byte = (0 << 2) | 1 | (0 << 5) = 0x01
    // next byte = offset low 8 bits = 4
    const input = new Uint8Array([
      8,           // uncompressed length = 8
      0x0C,        // literal: length = (0x0C >> 2) + 1 = 4
      65, 66, 67, 68, // "ABCD"
      0x01,        // copy-1: tagType=1, len=((0x01>>2)&7)+4=4, offset high=0
      0x04,        // offset low = 4
    ]);
    const output = decompressSnappy(input);
    expect(output.length).toBe(8);
    expect(new TextDecoder().decode(output)).toBe("ABCDABCD");
  });
});
