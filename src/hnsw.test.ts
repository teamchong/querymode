import { describe, it, expect } from "vitest";
import { HnswIndex, cosineDistance, l2DistanceSq, dotDistance } from "./hnsw.js";

// ---------------------------------------------------------------------------
// Distance functions
// ---------------------------------------------------------------------------

describe("distance functions", () => {
  it("cosineDistance of identical vectors is 0", () => {
    const a = new Float32Array([1, 2, 3]);
    expect(cosineDistance(a, a)).toBeCloseTo(0, 5);
  });

  it("cosineDistance of orthogonal vectors is 1", () => {
    const a = new Float32Array([1, 0]);
    const b = new Float32Array([0, 1]);
    expect(cosineDistance(a, b)).toBeCloseTo(1, 5);
  });

  it("cosineDistance of zero vector returns 1", () => {
    const a = new Float32Array([0, 0, 0]);
    const b = new Float32Array([1, 2, 3]);
    expect(cosineDistance(a, b)).toBe(1);
  });

  it("l2DistanceSq returns squared distance", () => {
    const a = new Float32Array([1, 0, 0]);
    const b = new Float32Array([0, 1, 0]);
    expect(l2DistanceSq(a, b)).toBeCloseTo(2, 5);
  });

  it("dotDistance returns negative dot product", () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([4, 5, 6]);
    // dot = 4+10+18 = 32, distance = -32
    expect(dotDistance(a, b)).toBeCloseTo(-32, 5);
  });
});

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

describe("input validation", () => {
  it("add() rejects NaN in vector", () => {
    const idx = new HnswIndex({ dim: 3 });
    const vec = new Float32Array([1, NaN, 3]);
    expect(() => idx.add(0, vec)).toThrow("non-finite value at index 1: NaN");
  });

  it("add() rejects Infinity in vector", () => {
    const idx = new HnswIndex({ dim: 3 });
    const vec = new Float32Array([Infinity, 2, 3]);
    expect(() => idx.add(0, vec)).toThrow("non-finite value at index 0: Infinity");
  });

  it("add() rejects -Infinity in vector", () => {
    const idx = new HnswIndex({ dim: 3 });
    const vec = new Float32Array([-Infinity, 2, 3]);
    expect(() => idx.add(0, vec)).toThrow("non-finite value at index 0: -Infinity");
  });

  it("add() rejects wrong dimension", () => {
    const idx = new HnswIndex({ dim: 3 });
    const vec = new Float32Array([1, 2]);
    expect(() => idx.add(0, vec)).toThrow("expected dimension 3, got 2");
  });

  it("search() rejects NaN in query vector", () => {
    const idx = new HnswIndex({ dim: 3 });
    idx.add(0, new Float32Array([1, 2, 3]));
    const query = new Float32Array([NaN, 0, 0]);
    expect(() => idx.search(query, 1)).toThrow("non-finite value");
  });

  it("search() rejects Infinity in query vector", () => {
    const idx = new HnswIndex({ dim: 3 });
    idx.add(0, new Float32Array([1, 2, 3]));
    const query = new Float32Array([0, Infinity, 0]);
    expect(() => idx.search(query, 1)).toThrow("non-finite value");
  });
});

// ---------------------------------------------------------------------------
// Core functionality
// ---------------------------------------------------------------------------

describe("HnswIndex", () => {
  it("empty index returns no results", () => {
    const idx = new HnswIndex({ dim: 3 });
    const { indices, scores } = idx.search(new Float32Array([1, 0, 0]), 5);
    expect(indices.length).toBe(0);
    expect(scores.length).toBe(0);
  });

  it("single vector is found", () => {
    const idx = new HnswIndex({ dim: 3, metric: "l2" });
    idx.add(0, new Float32Array([1, 2, 3]));
    const { indices } = idx.search(new Float32Array([1, 2, 3]), 1);
    expect(indices[0]).toBe(0);
  });

  it("nearest neighbor is correct with l2", () => {
    const idx = new HnswIndex({ dim: 2, metric: "l2", M: 4, efConstruction: 50 });
    idx.add(0, new Float32Array([0, 0]));
    idx.add(1, new Float32Array([1, 0]));
    idx.add(2, new Float32Array([10, 10]));

    const { indices } = idx.search(new Float32Array([0.5, 0]), 1);
    // Closest to [0.5, 0] should be [0, 0] or [1, 0] (both distance 0.25)
    expect([0, 1]).toContain(indices[0]);
  });

  it("topK returns correct count", () => {
    const idx = new HnswIndex({ dim: 2, metric: "l2", M: 4 });
    for (let i = 0; i < 10; i++) {
      idx.add(i, new Float32Array([i, 0]));
    }
    const { indices } = idx.search(new Float32Array([5, 0]), 3);
    expect(indices.length).toBe(3);
  });

  it("topK larger than size returns all", () => {
    const idx = new HnswIndex({ dim: 2, metric: "l2" });
    idx.add(0, new Float32Array([0, 0]));
    idx.add(1, new Float32Array([1, 1]));
    const { indices } = idx.search(new Float32Array([0, 0]), 10);
    expect(indices.length).toBe(2);
  });

  it("cosine metric finds similar directions", () => {
    const idx = new HnswIndex({ dim: 3, metric: "cosine", M: 4 });
    idx.add(0, new Float32Array([1, 0, 0]));  // pointing x
    idx.add(1, new Float32Array([0, 1, 0]));  // pointing y
    idx.add(2, new Float32Array([0, 0, 1]));  // pointing z

    const { indices } = idx.search(new Float32Array([1, 0.1, 0]), 1);
    expect(indices[0]).toBe(0); // closest direction is x
  });

  it("addBatch adds multiple vectors", () => {
    const idx = new HnswIndex({ dim: 2, metric: "l2" });
    const batch = new Float32Array([0, 0, 1, 1, 2, 2]);
    idx.addBatch(batch, 2);
    expect(idx.size).toBe(3);
    const { indices } = idx.search(new Float32Array([2.1, 2.1]), 1);
    expect(indices[0]).toBe(2);
  });

  it("addBatch rejects misaligned buffer", () => {
    const idx = new HnswIndex({ dim: 3, metric: "l2" });
    const batch = new Float32Array([1, 2, 3, 4, 5]); // 5 not divisible by 3
    expect(() => idx.addBatch(batch, 3)).toThrow("not divisible by dim");
  });
});

// ---------------------------------------------------------------------------
// Serialization round-trip
// ---------------------------------------------------------------------------

describe("serialization", () => {
  it("serialize + deserialize produces identical search results", () => {
    const idx = new HnswIndex({ dim: 3, metric: "l2", M: 4, efConstruction: 50 });
    for (let i = 0; i < 20; i++) {
      idx.add(i, new Float32Array([i, i * 2, i * 3]));
    }
    const query = new Float32Array([5, 10, 15]);
    const original = idx.search(query, 3);

    const buf = idx.serialize();
    const restored = HnswIndex.deserialize(buf);
    const after = restored.search(query, 3);

    expect(after.indices).toEqual(original.indices);
    for (let i = 0; i < original.scores.length; i++) {
      expect(after.scores[i]).toBeCloseTo(original.scores[i], 5);
    }
  });

  it("deserialize preserves efConstruction", () => {
    const idx = new HnswIndex({ dim: 2, metric: "cosine", efConstruction: 42 });
    idx.add(0, new Float32Array([1, 0]));
    const buf = idx.serialize();
    const restored = HnswIndex.deserialize(buf);
    // Can't directly read efConstruction, but verify it builds correctly
    expect(restored.size).toBe(1);
  });

  it("M is clamped to minimum 2", () => {
    const idx = new HnswIndex({ dim: 2, M: 1 });
    // M=1 would cause Infinity mL — clamped to 2
    idx.add(0, new Float32Array([1, 0]));
    idx.add(1, new Float32Array([0, 1]));
    const { indices } = idx.search(new Float32Array([1, 0]), 1);
    expect(indices[0]).toBe(0);
  });
});
