import { describe, it, expect } from "vitest";
import {
  initPartialAggState,
  updatePartialAgg,
  computePartialAgg,
  mergePartialAggs,
  finalizePartialAgg,
} from "./partial-agg.js";
import type { QueryDescriptor } from "./client.js";

describe("partial-agg", () => {
  it("initPartialAggState returns correct defaults", () => {
    const state = initPartialAggState("sum", "amount");
    expect(state.fn).toBe("sum");
    expect(state.column).toBe("amount");
    expect(state.sum).toBe(0);
    expect(state.count).toBe(0);
    expect(state.min).toBe(Infinity);
    expect(state.max).toBe(-Infinity);
  });

  it("updatePartialAgg accumulates values correctly", () => {
    const state = initPartialAggState("avg", "price");
    updatePartialAgg(state, 10);
    updatePartialAgg(state, 20);
    updatePartialAgg(state, 5);

    expect(state.sum).toBe(35);
    expect(state.count).toBe(3);
    expect(state.min).toBe(5);
    expect(state.max).toBe(20);
  });

  it("computePartialAgg computes sum over rows", () => {
    const rows = [{ amount: 10 }, { amount: 20 }, { amount: 30 }];
    const query: QueryDescriptor = {
      table: "orders",
      filters: [],
      projections: [],
      aggregates: [{ fn: "sum", column: "amount" }],
    };

    const result = computePartialAgg(rows, query);
    expect(result.states).toHaveLength(1);
    expect(result.states[0].sum).toBe(60);
    expect(result.states[0].count).toBe(3);
  });

  it("computePartialAgg with groupBy groups by key", () => {
    const rows = [
      { region: "us", amount: 10 },
      { region: "eu", amount: 20 },
      { region: "us", amount: 30 },
      { region: "eu", amount: 5 },
    ];
    const query: QueryDescriptor = {
      table: "orders",
      filters: [],
      projections: [],
      aggregates: [{ fn: "sum", column: "amount" }],
      groupBy: ["region"],
    };

    const result = computePartialAgg(rows, query);
    expect(result.groups).toBeDefined();
    expect(result.groups!.get("us")![0].sum).toBe(40);
    expect(result.groups!.get("eu")![0].sum).toBe(25);
  });

  it("mergePartialAggs merges two ungrouped partials", () => {
    const p1 = computePartialAgg([{ amount: 10 }, { amount: 20 }], {
      table: "t",
      filters: [],
      projections: [],
      aggregates: [{ fn: "sum", column: "amount" }],
    });
    const p2 = computePartialAgg([{ amount: 5 }, { amount: 100 }], {
      table: "t",
      filters: [],
      projections: [],
      aggregates: [{ fn: "sum", column: "amount" }],
    });

    const merged = mergePartialAggs([p1, p2]);
    expect(merged.states[0].sum).toBe(135);
    expect(merged.states[0].count).toBe(4);
    expect(merged.states[0].min).toBe(5);
    expect(merged.states[0].max).toBe(100);
  });

  it("stddev/variance return null for single-element groups (SQL STDDEV_SAMP)", () => {
    const query: QueryDescriptor = {
      table: "t",
      filters: [],
      projections: [],
      aggregates: [
        { fn: "stddev", column: "v" },
        { fn: "variance", column: "v" },
      ],
    };
    // Single value → count=1 → STDDEV_SAMP/VAR_SAMP undefined → null
    const partial = computePartialAgg([{ v: 42 }], query);
    const result = finalizePartialAgg(partial, query);
    expect(result[0]["stddev_v"]).toBe(null);
    expect(result[0]["variance_v"]).toBe(null);
  });

  it("stddev/variance return values for multi-element groups", () => {
    const query: QueryDescriptor = {
      table: "t",
      filters: [],
      projections: [],
      aggregates: [
        { fn: "stddev", column: "v" },
        { fn: "variance", column: "v" },
      ],
    };
    // [10, 20, 30] → mean=20, population variance=66.67, stddev=8.16
    const partial = computePartialAgg([{ v: 10 }, { v: 20 }, { v: 30 }], query);
    const result = finalizePartialAgg(partial, query);
    expect(result[0]["variance_v"]).toBeCloseTo(66.667, 1);
    expect(result[0]["stddev_v"]).toBeCloseTo(8.165, 1);
  });

  it("finalizePartialAgg computes avg and other aggregates", () => {
    const rows = [{ v: 10 }, { v: 20 }, { v: 30 }];
    const query: QueryDescriptor = {
      table: "t",
      filters: [],
      projections: [],
      aggregates: [
        { fn: "avg", column: "v" },
        { fn: "sum", column: "v" },
        { fn: "min", column: "v" },
        { fn: "max", column: "v" },
        { fn: "count", column: "v" },
      ],
    };

    const partial = computePartialAgg(rows, query);
    const result = finalizePartialAgg(partial, query);

    expect(result).toHaveLength(1);
    expect(result[0]["avg_v"]).toBe(20);
    expect(result[0]["sum_v"]).toBe(60);
    expect(result[0]["min_v"]).toBe(10);
    expect(result[0]["max_v"]).toBe(30);
    expect(result[0]["count_v"]).toBe(3);
  });
});
