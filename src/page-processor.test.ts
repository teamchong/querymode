import { describe, it, expect } from "vitest";
import { createPageProcessor } from "./page-processor.js";

describe("PageProcessor (TS fallback)", () => {
  const proc = createPageProcessor(null);

  it("sumFloat64 sums elements correctly", () => {
    const f64 = new Float64Array([1.5, 2.5, 3.0]);
    expect(proc.sumFloat64(f64.buffer)).toBe(7.0);
  });

  it("minFloat64 finds minimum", () => {
    const f64 = new Float64Array([5.0, 2.0, 8.0, 1.0]);
    expect(proc.minFloat64(f64.buffer)).toBe(1.0);
  });

  it("maxFloat64 finds maximum", () => {
    const f64 = new Float64Array([5.0, 2.0, 8.0, 1.0]);
    expect(proc.maxFloat64(f64.buffer)).toBe(8.0);
  });

  it("decodePage decodes int32 correctly", () => {
    const i32 = new Int32Array([10, 20, 30]);
    const result = proc.decodePage(i32.buffer, "int32");
    expect(result).toEqual([10, 20, 30]);
  });

  it("handles empty buffer", () => {
    const empty = new ArrayBuffer(0);
    expect(proc.sumFloat64(empty)).toBe(0);
    expect(proc.minFloat64(empty)).toBe(Infinity);
    expect(proc.maxFloat64(empty)).toBe(-Infinity);
  });

  it("vectorSearchPage returns top-K by cosine similarity", () => {
    const dim = 3;
    // 4 vectors: [1,0,0], [0,1,0], [0.9,0.1,0], [0,0,1]
    const vectors = new Float32Array([
      1, 0, 0,
      0, 1, 0,
      0.9, 0.1, 0,
      0, 0, 1,
    ]);
    const query = new Float32Array([1, 0, 0]);
    const { indices, scores } = proc.vectorSearchPage(vectors, 4, dim, query, 2);

    expect(indices.length).toBe(2);
    expect(scores.length).toBe(2);
    // Vector 0 ([1,0,0]) is most similar to query [1,0,0]
    expect(indices[0]).toBe(0);
    // Vector 2 ([0.9,0.1,0]) is second most similar
    expect(indices[1]).toBe(2);
    // Score for exact match should be ~1.0
    expect(scores[0]).toBeCloseTo(1.0, 5);
    // Score for near-match should be high but < 1.0
    expect(scores[1]).toBeGreaterThan(0.9);
    expect(scores[1]).toBeLessThan(1.0);
  });
});
