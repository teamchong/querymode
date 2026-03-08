import { describe, it, expect } from "vitest";
import { descriptorToCode } from "./descriptor-to-code.js";
import type { QueryDescriptor } from "./client.js";

function makeDesc(overrides: Partial<QueryDescriptor> = {}): QueryDescriptor {
  return {
    table: "orders",
    filters: [],
    projections: [],
    ...overrides,
  };
}

describe("descriptorToCode", () => {
  it("minimal query — table only", () => {
    const code = descriptorToCode(makeDesc());
    expect(code).toContain('.table("orders")');
    expect(code).toContain(".collect()");
    expect(code).toMatch(/^const result = await /);
  });

  it("custom variable name and table function", () => {
    const code = descriptorToCode(makeDesc(), { variableName: "data", tableFn: "db" });
    expect(code).toMatch(/^const data = await db/);
  });

  it("eq/neq/gt/gte/lt/lte filters use .filter()", () => {
    const code = descriptorToCode(makeDesc({
      filters: [
        { column: "amount", op: "gt", value: 100 },
        { column: "status", op: "eq", value: "active" },
        { column: "score", op: "lte", value: 99.5 },
      ],
    }));
    expect(code).toContain('.filter("amount", "gt", 100)');
    expect(code).toContain('.filter("status", "eq", "active")');
    expect(code).toContain('.filter("score", "lte", 99.5)');
  });

  it("is_null/is_not_null filters use shorthand", () => {
    const code = descriptorToCode(makeDesc({
      filters: [
        { column: "email", op: "is_null", value: 0 },
        { column: "name", op: "is_not_null", value: 0 },
      ],
    }));
    expect(code).toContain('.whereNull("email")');
    expect(code).toContain('.whereNotNull("name")');
  });

  it("in/not_in filters use shorthand", () => {
    const code = descriptorToCode(makeDesc({
      filters: [
        { column: "region", op: "in", value: ["us", "eu", "ap"] },
        { column: "status", op: "not_in", value: [1, 2] },
      ],
    }));
    expect(code).toContain('.whereIn("region", ["us", "eu", "ap"])');
    expect(code).toContain('.whereNotIn("status", [1, 2])');
  });

  it("between/not_between filters use shorthand", () => {
    const code = descriptorToCode(makeDesc({
      filters: [
        { column: "age", op: "between", value: [18, 65] },
        { column: "score", op: "not_between", value: [0, 10] },
      ],
    }));
    expect(code).toContain('.whereBetween("age", 18, 65)');
    expect(code).toContain('.whereNotBetween("score", 0, 10)');
  });

  it("like/not_like filters use shorthand", () => {
    const code = descriptorToCode(makeDesc({
      filters: [
        { column: "name", op: "like", value: "%alice%" },
        { column: "email", op: "not_like", value: "%spam%" },
      ],
    }));
    expect(code).toContain('.whereLike("name", "%alice%")');
    expect(code).toContain('.whereNotLike("email", "%spam%")');
  });

  it("filterGroups (OR logic)", () => {
    const code = descriptorToCode(makeDesc({
      filterGroups: [
        [{ column: "dept", op: "eq", value: "eng" }],
        [{ column: "age", op: "gt", value: 30 }],
      ],
    }));
    expect(code).toContain(".whereOr(");
    expect(code).toContain('column: "dept"');
    expect(code).toContain('column: "age"');
  });

  it("projections", () => {
    const code = descriptorToCode(makeDesc({
      projections: ["id", "name", "email"],
    }));
    expect(code).toContain('.select("id", "name", "email")');
  });

  it("sort ascending (default)", () => {
    const code = descriptorToCode(makeDesc({
      sortColumn: "created_at",
      sortDirection: "asc",
    }));
    expect(code).toContain('.sort("created_at")');
    expect(code).not.toContain('"asc"');
  });

  it("sort descending", () => {
    const code = descriptorToCode(makeDesc({
      sortColumn: "amount",
      sortDirection: "desc",
    }));
    expect(code).toContain('.sort("amount", "desc")');
  });

  it("limit and offset", () => {
    const code = descriptorToCode(makeDesc({
      limit: 20,
      offset: 100,
    }));
    expect(code).toContain(".limit(20)");
    expect(code).toContain(".offset(100)");
  });

  it("aggregates with groupBy", () => {
    const code = descriptorToCode(makeDesc({
      groupBy: ["region", "category"],
      aggregates: [
        { fn: "sum", column: "amount", alias: "total" },
        { fn: "count", column: "*" },
        { fn: "avg", column: "score", alias: "avg_score" },
      ],
    }));
    expect(code).toContain('.groupBy("region", "category")');
    expect(code).toContain('.aggregate("sum", "amount", "total")');
    expect(code).toContain('.aggregate("count", "*")');
    expect(code).toContain('.aggregate("avg", "score", "avg_score")');
  });

  it("percentile aggregate", () => {
    const code = descriptorToCode(makeDesc({
      aggregates: [
        { fn: "percentile", column: "latency", alias: "p99", percentileTarget: 0.99 },
      ],
    }));
    expect(code).toContain('.percentile("latency", 0.99, "p99")');
  });

  it("distinct with columns", () => {
    const code = descriptorToCode(makeDesc({
      distinct: ["region", "category"],
    }));
    expect(code).toContain('.distinct("region", "category")');
  });

  it("distinct without columns", () => {
    const code = descriptorToCode(makeDesc({
      distinct: [],
    }));
    expect(code).toContain(".distinct()");
  });

  it("version", () => {
    const code = descriptorToCode(makeDesc({ version: 3 }));
    expect(code).toContain(".version(3)");
  });

  it("cache TTL", () => {
    const code = descriptorToCode(makeDesc({ cacheTTL: 60000 }));
    expect(code).toContain(".cache({ ttl: 60000 })");
  });

  it("window function", () => {
    const code = descriptorToCode(makeDesc({
      windows: [{
        fn: "row_number",
        partitionBy: ["region"],
        orderBy: [{ column: "amount", direction: "desc" }],
        alias: "rank",
      }],
    }));
    expect(code).toContain(".window(");
    expect(code).toContain('fn: "row_number"');
    expect(code).toContain('partitionBy: ["region"]');
    expect(code).toContain('alias: "rank"');
  });

  it("window function with column, args, and frame", () => {
    const code = descriptorToCode(makeDesc({
      windows: [{
        fn: "lag",
        column: "price",
        partitionBy: ["symbol"],
        orderBy: [{ column: "ts", direction: "asc" }],
        alias: "prev_price",
        args: { offset: 1, default_: 0 },
        frame: { type: "rows", start: "unbounded", end: "current" },
      }],
    }));
    expect(code).toContain('column: "price"');
    expect(code).toContain("offset: 1");
    expect(code).toContain("default_: 0");
    expect(code).toContain('type: "rows"');
    expect(code).toContain('start: "unbounded"');
    expect(code).toContain('end: "current"');
  });

  it("join", () => {
    const code = descriptorToCode(makeDesc({
      join: {
        right: makeDesc({ table: "users", filters: [{ column: "active", op: "eq", value: true as unknown as number }] }),
        leftKey: "user_id",
        rightKey: "id",
        type: "left",
      },
    }));
    expect(code).toContain(".join(");
    expect(code).toContain('.table("users")');
    expect(code).toContain('left: "user_id"');
    expect(code).toContain('right: "id"');
    expect(code).toContain('"left"');
  });

  it("join — inner (default) omits type", () => {
    const code = descriptorToCode(makeDesc({
      join: {
        right: makeDesc({ table: "users" }),
        leftKey: "user_id",
        rightKey: "id",
        type: "inner",
      },
    }));
    expect(code).toContain(".join(");
    expect(code).not.toContain('"inner"');
  });

  it("set operation — union", () => {
    const code = descriptorToCode(makeDesc({
      setOperation: { mode: "union", right: makeDesc({ table: "archive_orders" }) },
    }));
    expect(code).toContain(".union(");
    expect(code).toContain('.table("archive_orders")');
  });

  it("set operation — union all", () => {
    const code = descriptorToCode(makeDesc({
      setOperation: { mode: "union_all", right: makeDesc({ table: "archive" }) },
    }));
    expect(code).toContain(".union(");
    expect(code).toContain("true");
  });

  it("set operation — intersect", () => {
    const code = descriptorToCode(makeDesc({
      setOperation: { mode: "intersect", right: makeDesc({ table: "vip_orders" }) },
    }));
    expect(code).toContain(".intersect(");
  });

  it("set operation — except", () => {
    const code = descriptorToCode(makeDesc({
      setOperation: { mode: "except", right: makeDesc({ table: "cancelled" }) },
    }));
    expect(code).toContain(".except(");
  });

  it("computed columns emit identity function since callbacks are not serializable", () => {
    const code = descriptorToCode(makeDesc({
      computedColumns: [{ alias: "total", fn: (row) => (row.price as number) * (row.qty as number) }],
    }));
    expect(code).toContain('.computed("total"');
    expect(code).toContain("(row) =>");
  });

  it("vector search", () => {
    const code = descriptorToCode(makeDesc({
      vectorSearch: {
        column: "embedding",
        queryVector: new Float32Array([0.1, 0.2, 0.3]),
        topK: 10,
        metric: "l2",
        nprobe: 16,
      },
    }));
    expect(code).toContain('.vector("embedding"');
    expect(code).toContain("new Float32Array(");
    expect(code).toContain("10");
    expect(code).toContain('metric: "l2"');
    expect(code).toContain("nprobe: 16");
  });

  it("vector search with default metric omits it", () => {
    const code = descriptorToCode(makeDesc({
      vectorSearch: {
        column: "emb",
        queryVector: new Float32Array([1, 2]),
        topK: 5,
      },
    }));
    expect(code).toContain('.vector("emb"');
    expect(code).not.toContain("metric:");
  });

  it("bigint filter values use n suffix", () => {
    const code = descriptorToCode(makeDesc({
      filters: [{ column: "id", op: "eq", value: 9007199254740993n }],
    }));
    expect(code).toContain("9007199254740993n");
  });

  it("bigint in array values", () => {
    const code = descriptorToCode(makeDesc({
      filters: [{ column: "id", op: "in", value: [1n, 2n, 3n] }],
    }));
    expect(code).toContain("[1n, 2n, 3n]");
  });

  it("complex real-world query", () => {
    const code = descriptorToCode(makeDesc({
      filters: [
        { column: "category", op: "eq", value: "Electronics" },
        { column: "amount", op: "gte", value: 100 },
      ],
      projections: ["id", "amount", "region"],
      sortColumn: "amount",
      sortDirection: "desc",
      limit: 20,
    }));
    expect(code).toBe(
      `const result = await qm\n` +
      `  .table("orders")\n` +
      `  .filter("category", "eq", "Electronics")\n` +
      `  .filter("amount", "gte", 100)\n` +
      `  .select("id", "amount", "region")\n` +
      `  .sort("amount", "desc")\n` +
      `  .limit(20)\n` +
      `  .collect()`
    );
  });

  it("order: filters before projections before sort before limit", () => {
    const code = descriptorToCode(makeDesc({
      filters: [{ column: "x", op: "gt", value: 0 }],
      projections: ["x", "y"],
      sortColumn: "x",
      sortDirection: "asc",
      offset: 5,
      limit: 10,
    }));
    const filterIdx = code.indexOf(".filter(");
    const selectIdx = code.indexOf(".select(");
    const sortIdx = code.indexOf(".sort(");
    const offsetIdx = code.indexOf(".offset(");
    const limitIdx = code.indexOf(".limit(");
    const collectIdx = code.indexOf(".collect()");
    expect(filterIdx).toBeLessThan(selectIdx);
    expect(selectIdx).toBeLessThan(sortIdx);
    expect(sortIdx).toBeLessThan(offsetIdx);
    expect(offsetIdx).toBeLessThan(limitIdx);
    expect(limitIdx).toBeLessThan(collectIdx);
  });

  it("pipe stages emit identity pipe calls", () => {
    const code = descriptorToCode(makeDesc({
      pipeStages: [((u: unknown) => u) as never, ((u: unknown) => u) as never],
    }));
    const pipeMatches = code.match(/\.pipe\(/g);
    expect(pipeMatches).toHaveLength(2);
  });

  it("subqueryIn", () => {
    const code = descriptorToCode(makeDesc({
      subqueryIn: [{ column: "region", valueSet: new Set(["us", "eu"]) }],
    }));
    expect(code).toContain('.filterIn("region"');
    expect(code).toContain('"us"');
    expect(code).toContain('"eu"');
  });

  it("limit 0 is preserved", () => {
    const code = descriptorToCode(makeDesc({ limit: 0 }));
    expect(code).toContain(".limit(0)");
  });

  it("offset 0 is NOT emitted (falsy)", () => {
    const code = descriptorToCode(makeDesc({ offset: 0 }));
    expect(code).not.toContain(".offset(");
  });
});

describe("DataFrame.toCode()", () => {
  it("instance method produces same output as static", async () => {
    // Import DataFrame to test the instance method
    const { DataFrame } = await import("./client.js");

    const noopExecutor = {
      async execute() { return { rows: [], durationMs: 0, bytesRead: 0, rowsScanned: 0 }; },
      async explain() { return { plan: [], fragmentsTotal: 0, fragmentsScanned: 0, pagesTotal: 0, pagesScanned: 0 }; },
    };

    const df = new DataFrame("events", noopExecutor)
      .filter("type", "eq", "click")
      .select("id", "ts")
      .sort("ts", "desc")
      .limit(50);

    const code = df.toCode();
    const expected = descriptorToCode(df.toDescriptor());
    expect(code).toBe(expected);
    expect(code).toContain('.table("events")');
    expect(code).toContain('.filter("type", "eq", "click")');
    expect(code).toContain('.select("id", "ts")');
    expect(code).toContain('.sort("ts", "desc")');
    expect(code).toContain(".limit(50)");
  });

  it("static descriptorToCode is accessible", async () => {
    const { DataFrame } = await import("./client.js");
    expect(DataFrame.descriptorToCode).toBe(descriptorToCode);
  });
});
