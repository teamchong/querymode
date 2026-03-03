import { describe, it, expect } from "vitest";
import { decodePage } from "./decode.js";

describe("decodePage", () => {
  it("decodes int32 correctly", () => {
    const i32 = new Int32Array([10, 20, 30]);
    expect(decodePage(i32.buffer, "int32")).toEqual([10, 20, 30]);
  });

  it("decodes float64 correctly", () => {
    const f64 = new Float64Array([1.5, 2.5, 3.0]);
    expect(decodePage(f64.buffer, "float64")).toEqual([1.5, 2.5, 3.0]);
  });

  it("decodes utf8 correctly", () => {
    const enc = new TextEncoder();
    const str = "hello";
    const strBytes = enc.encode(str);
    const buf = new ArrayBuffer(4 + strBytes.length);
    new DataView(buf).setUint32(0, strBytes.length, true);
    new Uint8Array(buf, 4).set(strBytes);
    expect(decodePage(buf, "utf8")).toEqual(["hello"]);
  });

  it("handles empty buffer", () => {
    expect(decodePage(new ArrayBuffer(0), "int32")).toEqual([]);
    expect(decodePage(new ArrayBuffer(0), "float64")).toEqual([]);
  });
});
