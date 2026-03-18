/**
 * HTTP API + query schema tests.
 *
 * Part 1: HTTP routing tests using SELF.fetch for stateless endpoints only
 * (health, 404 — no DO interaction, no storage mutation).
 *
 * Part 2: Schema validation tests using parseAndValidateQuery directly
 * (covers orderBy alias, aggregate fns, select alias, groupBy, etc.)
 *
 * Data-dependent query tests are in operators-conformance.test.ts (at scale
 * against DuckDB) and convenience.test.ts (in-memory).
 */
import { describe, it, expect } from "vitest";
import { SELF } from "cloudflare:test";
import { parseAndValidateQuery } from "./query-schema.js";

// ── HTTP routing (stateless endpoints only) ─────────────────────────────────

describe("HTTP routing", () => {
  it("GET /health returns ok", async () => {
    const res = await SELF.fetch("https://fake-host/health");
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toBe("application/json");
    expect(res.headers.get("x-querymode-request-id")).toBeDefined();
    const body = await res.json() as Record<string, unknown>;
    expect(body.status).toBe("ok");
    expect(body.service).toBe("querymode");
    expect(body.timestamp).toBeDefined();
  });

  it("GET /nonexistent returns 404", async () => {
    const res = await SELF.fetch("https://fake-host/nonexistent");
    expect(res.status).toBe(404);
    const text = await res.text();
    expect(text).toContain("/query");
  });
});

// ── Schema validation ─────────────────────────────────────────────────────

describe("parseAndValidateQuery", () => {
  it("parses minimal query", () => {
    const q = parseAndValidateQuery({ table: "events" });
    expect(q.table).toBe("events");
    expect(q.filters).toEqual([]);
    expect(q.projections).toEqual([]);
  });

  it("parses filter with all ops", () => {
    for (const op of ["eq", "neq", "gt", "gte", "lt", "lte", "in"]) {
      const value = op === "in" ? [1, 2] : 42;
      const q = parseAndValidateQuery({
        table: "t",
        filters: [{ column: "x", op, value }],
      });
      expect(q.filters[0].op).toBe(op);
    }
  });

  it("parses sortColumn + sortDirection", () => {
    const q = parseAndValidateQuery({
      table: "t",
      sortColumn: "amount",
      sortDirection: "desc",
    });
    expect(q.sortColumn).toBe("amount");
    expect(q.sortDirection).toBe("desc");
  });

  it("parses orderBy alias → sortColumn/sortDirection", () => {
    const q = parseAndValidateQuery({
      table: "t",
      orderBy: { column: "amount", desc: true },
    });
    expect(q.sortColumn).toBe("amount");
    expect(q.sortDirection).toBe("desc");
  });

  it("orderBy without desc defaults to asc", () => {
    const q = parseAndValidateQuery({
      table: "t",
      orderBy: { column: "x" },
    });
    expect(q.sortColumn).toBe("x");
    expect(q.sortDirection).toBe("asc");
  });

  it("sortColumn takes precedence over orderBy", () => {
    const q = parseAndValidateQuery({
      table: "t",
      sortColumn: "a",
      sortDirection: "desc",
      orderBy: { column: "b" },
    });
    expect(q.sortColumn).toBe("a");
    expect(q.sortDirection).toBe("desc");
  });

  it("parses select alias → projections", () => {
    const q = parseAndValidateQuery({
      table: "t",
      select: ["id", "name"],
    });
    expect(q.projections).toEqual(["id", "name"]);
  });

  it("projections takes precedence over select", () => {
    const q = parseAndValidateQuery({
      table: "t",
      projections: ["a"],
      select: ["b"],
    });
    expect(q.projections).toEqual(["a"]);
  });

  it("parses limit + offset", () => {
    const q = parseAndValidateQuery({ table: "t", limit: 10, offset: 5 });
    expect(q.limit).toBe(10);
    expect(q.offset).toBe(5);
  });

  it("parses groupBy", () => {
    const q = parseAndValidateQuery({
      table: "t",
      groupBy: ["region", "category"],
    });
    expect(q.groupBy).toEqual(["region", "category"]);
  });

  it("parses cacheTTL", () => {
    const q = parseAndValidateQuery({ table: "t", cacheTTL: 60 });
    expect(q.cacheTTL).toBe(60);
  });

  // Aggregate functions
  for (const fn of ["sum", "avg", "min", "max", "count", "count_distinct", "stddev", "variance", "median", "percentile"]) {
    it(`accepts aggregate fn="${fn}"`, () => {
      const q = parseAndValidateQuery({
        table: "t",
        aggregates: [{ fn, column: "x" }],
      });
      expect(q.aggregates).toHaveLength(1);
      expect(q.aggregates![0].fn).toBe(fn);
    });
  }

  it("accepts percentileTarget", () => {
    const q = parseAndValidateQuery({
      table: "t",
      aggregates: [{ fn: "percentile", column: "x", percentileTarget: 0.95 }],
    });
    expect(q.aggregates![0].percentileTarget).toBe(0.95);
  });

  it("accepts aggregate alias", () => {
    const q = parseAndValidateQuery({
      table: "t",
      aggregates: [{ fn: "sum", column: "amount", alias: "total" }],
    });
    expect(q.aggregates![0].alias).toBe("total");
  });

  // Rejection cases
  it("rejects empty table name", () => {
    expect(() => parseAndValidateQuery({ table: "" })).toThrow(/table/i);
  });

  it("rejects missing table", () => {
    expect(() => parseAndValidateQuery({})).toThrow();
  });

  it("rejects invalid filter op", () => {
    expect(() => parseAndValidateQuery({
      table: "t",
      filters: [{ column: "x", op: "INVALID", value: 1 }],
    })).toThrow();
  });

  it("rejects invalid aggregate fn", () => {
    expect(() => parseAndValidateQuery({
      table: "t",
      aggregates: [{ fn: "BOGUS", column: "x" }],
    })).toThrow();
  });

  it("rejects empty filter column", () => {
    expect(() => parseAndValidateQuery({
      table: "t",
      filters: [{ column: "", op: "eq", value: 1 }],
    })).toThrow();
  });

  it("rejects negative limit", () => {
    expect(() => parseAndValidateQuery({ table: "t", limit: -1 })).toThrow();
  });

  it("rejects negative offset", () => {
    expect(() => parseAndValidateQuery({ table: "t", offset: -5 })).toThrow();
  });

  it("rejects percentileTarget out of range", () => {
    expect(() => parseAndValidateQuery({
      table: "t",
      aggregates: [{ fn: "percentile", column: "x", percentileTarget: 1.5 }],
    })).toThrow();
  });

  it("accepts zero limit (returns empty set)", () => {
    const q = parseAndValidateQuery({ table: "t", limit: 0 });
    expect(q.limit).toBe(0);
  });

  it("rejects float limit", () => {
    expect(() => parseAndValidateQuery({ table: "t", limit: 1.5 })).toThrow();
  });

  it("rejects float offset", () => {
    expect(() => parseAndValidateQuery({ table: "t", offset: 2.5 })).toThrow();
  });

  it("accepts multiple filters", () => {
    const q = parseAndValidateQuery({
      table: "t",
      filters: [
        { column: "category", op: "eq", value: "alpha" },
        { column: "value", op: "gt", value: 900 },
        { column: "id", op: "gte", value: 50000 },
      ],
    });
    expect(q.filters).toHaveLength(3);
    expect(q.filters[0].op).toBe("eq");
    expect(q.filters[1].op).toBe("gt");
    expect(q.filters[2].op).toBe("gte");
  });

  it("accepts multiple aggregates", () => {
    const q = parseAndValidateQuery({
      table: "t",
      aggregates: [
        { fn: "count", column: "*" },
        { fn: "avg", column: "value", alias: "avg_val" },
        { fn: "sum", column: "value", alias: "total" },
      ],
    });
    expect(q.aggregates).toHaveLength(3);
    expect(q.aggregates![0].fn).toBe("count");
    expect(q.aggregates![1].alias).toBe("avg_val");
    expect(q.aggregates![2].alias).toBe("total");
  });

  it("accepts groupBy with aggregates", () => {
    const q = parseAndValidateQuery({
      table: "t",
      groupBy: ["category"],
      aggregates: [{ fn: "count", column: "*" }],
    });
    expect(q.groupBy).toEqual(["category"]);
    expect(q.aggregates).toHaveLength(1);
  });

  it("accepts IN filter with string array", () => {
    const q = parseAndValidateQuery({
      table: "t",
      filters: [{ column: "cat", op: "in", value: ["alpha", "beta"] }],
    });
    expect(q.filters[0].value).toEqual(["alpha", "beta"]);
  });

  it("accepts aggregates + limit (limit applies to output)", () => {
    const q = parseAndValidateQuery({
      table: "t",
      aggregates: [{ fn: "count", column: "*" }],
      limit: 1,
    });
    expect(q.aggregates).toHaveLength(1);
    expect(q.limit).toBe(1);
  });

  it("accepts orderBy + limit + select together", () => {
    const q = parseAndValidateQuery({
      table: "t",
      orderBy: { column: "value", desc: true },
      limit: 5,
      select: ["id", "value"],
    });
    expect(q.sortColumn).toBe("value");
    expect(q.sortDirection).toBe("desc");
    expect(q.limit).toBe(5);
    expect(q.projections).toEqual(["id", "value"]);
  });

  it("accepts large offset for pagination", () => {
    const q = parseAndValidateQuery({ table: "t", offset: 99995, limit: 10 });
    expect(q.offset).toBe(99995);
    expect(q.limit).toBe(10);
  });

  it("rejects empty aggregate column", () => {
    expect(() => parseAndValidateQuery({
      table: "t",
      aggregates: [{ fn: "sum", column: "" }],
    })).toThrow();
  });

  it("accepts is_null filter without value", () => {
    const q = parseAndValidateQuery({
      table: "t",
      filters: [{ column: "name", op: "is_null" }],
    });
    expect(q.filters[0].op).toBe("is_null");
    expect(q.filters[0].value).toBeUndefined();
  });

  it("accepts is_not_null filter without value", () => {
    const q = parseAndValidateQuery({
      table: "t",
      filters: [{ column: "name", op: "is_not_null" }],
    });
    expect(q.filters[0].op).toBe("is_not_null");
  });

  it("accepts distinct columns", () => {
    const q = parseAndValidateQuery({
      table: "t",
      distinct: ["region", "category"],
    });
    expect(q.distinct).toEqual(["region", "category"]);
  });
});
