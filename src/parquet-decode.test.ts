import { describe, it, expect } from "vitest";
import { decompressSnappy, decodeParquetColumnChunk } from "./parquet-decode.js";

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
