import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { deriveQueryCost } from "../client.js";
import type { ExplainResult } from "../types.js";

describe("deriveQueryCost", () => {
  it("rates trivial queries", () => {
    const plan = makePlan({ estimatedRows: 100, estimatedBytes: 5000, estimatedR2Reads: 1, fragments: 1 });
    const cost = deriveQueryCost(plan);
    expect(cost.rating).toBe("trivial");
    expect(cost.fanOut).toBe(false);
    expect(cost.estimatedDOs).toBe(0);
  });

  it("rates light queries", () => {
    const cost = deriveQueryCost(makePlan({ estimatedRows: 50_000 }));
    expect(cost.rating).toBe("light");
  });

  it("rates moderate queries", () => {
    const cost = deriveQueryCost(makePlan({ estimatedRows: 500_000 }));
    expect(cost.rating).toBe("moderate");
  });

  it("rates heavy queries", () => {
    const cost = deriveQueryCost(makePlan({ estimatedRows: 5_000_000 }));
    expect(cost.rating).toBe("heavy");
  });

  it("rates extreme queries", () => {
    const cost = deriveQueryCost(makePlan({ estimatedRows: 50_000_000 }));
    expect(cost.rating).toBe("extreme");
  });

  it("upgrades to extreme for large byte reads", () => {
    const cost = deriveQueryCost(makePlan({ estimatedRows: 100_000, estimatedBytes: 600_000_000 }));
    expect(cost.rating).toBe("extreme");
  });

  it("detects fan-out conditions", () => {
    const cost = deriveQueryCost(makePlan({ estimatedRows: 200_000, fragments: 5, fragmentsScanned: 5 }));
    expect(cost.fanOut).toBe(true);
    expect(cost.estimatedDOs).toBe(5);
  });

  it("detects hierarchical reduction", () => {
    const cost = deriveQueryCost(makePlan({ estimatedRows: 500_000, fragments: 100, fragmentsScanned: 100 }));
    expect(cost.hierarchicalReduction).toBe(true);
    expect(cost.rating).toBe("heavy"); // upgraded from moderate
  });

  it("no fan-out for small datasets", () => {
    const cost = deriveQueryCost(makePlan({ estimatedRows: 50_000, fragments: 1 }));
    expect(cost.fanOut).toBe(false);
    expect(cost.estimatedDOs).toBe(0);
  });
});

// maxScanRows E2E tests live in search-e2e.test.ts (shares WASM-loaded executor).

// Helper to create a minimal ExplainResult
function makePlan(overrides: Partial<ExplainResult>): ExplainResult {
  return {
    table: "test",
    format: "lance",
    totalRows: overrides.estimatedRows ?? 1000,
    columns: [],
    pagesTotal: 10,
    pagesSkipped: 0,
    pagesScanned: 10,
    estimatedBytes: overrides.estimatedBytes ?? 10_000,
    estimatedR2Reads: overrides.estimatedR2Reads ?? 5,
    fragments: overrides.fragments ?? 1,
    fragmentsSkipped: (overrides.fragments ?? 1) - (overrides.fragmentsScanned ?? overrides.fragments ?? 1),
    fragmentsScanned: overrides.fragmentsScanned ?? overrides.fragments ?? 1,
    filters: [],
    metaCached: false,
    estimatedRows: overrides.estimatedRows ?? 1000,
    ...overrides,
  };
}
