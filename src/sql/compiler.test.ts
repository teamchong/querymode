import { describe, it, expect } from "vitest";
import { compile, compileFull } from "./compiler.js";
import { parse } from "./parser.js";

function sql(query: string) {
  return compile(parse(query));
}

function sqlFull(query: string) {
  return compileFull(parse(query));
}

describe("SQL Compiler", () => {
  it("compiles SELECT * FROM table", () => {
    const desc = sql("SELECT * FROM users");
    expect(desc.table).toBe("users");
    expect(desc.projections).toEqual([]);
    expect(desc.filters).toEqual([]);
  });

  it("compiles SELECT with columns", () => {
    const desc = sql("SELECT name, age FROM users");
    expect(desc.projections).toEqual(["name", "age"]);
  });

  it("compiles WHERE with equality", () => {
    const desc = sql("SELECT * FROM t WHERE status = 'active'");
    expect(desc.filters).toEqual([{ column: "status", op: "eq", value: "active" }]);
  });

  it("compiles WHERE with comparison operators", () => {
    const desc = sql("SELECT * FROM t WHERE age > 25 AND score <= 100");
    expect(desc.filters).toHaveLength(2);
    expect(desc.filters[0]).toEqual({ column: "age", op: "gt", value: 25 });
    expect(desc.filters[1]).toEqual({ column: "score", op: "lte", value: 100 });
  });

  it("compiles WHERE != as neq", () => {
    const desc = sql("SELECT * FROM t WHERE x != 0");
    expect(desc.filters[0].op).toBe("neq");
  });

  it("compiles IS NULL", () => {
    const desc = sql("SELECT * FROM t WHERE col IS NULL");
    expect(desc.filters).toEqual([{ column: "col", op: "is_null", value: 0 }]);
  });

  it("compiles IS NOT NULL", () => {
    const desc = sql("SELECT * FROM t WHERE col IS NOT NULL");
    expect(desc.filters).toEqual([{ column: "col", op: "is_not_null", value: 0 }]);
  });

  it("compiles IN list", () => {
    const desc = sql("SELECT * FROM t WHERE id IN (1, 2, 3)");
    expect(desc.filters).toEqual([{ column: "id", op: "in", value: [1, 2, 3] }]);
  });

  it("compiles BETWEEN as single between filter", () => {
    const desc = sql("SELECT * FROM t WHERE age BETWEEN 18 AND 65");
    expect(desc.filters).toEqual([
      { column: "age", op: "between", value: [18, 65] },
    ]);
  });

  it("compiles ORDER BY", () => {
    const desc = sql("SELECT * FROM t ORDER BY amount DESC");
    expect(desc.sortColumn).toBe("amount");
    expect(desc.sortDirection).toBe("desc");
  });

  it("compiles LIMIT and OFFSET", () => {
    const desc = sql("SELECT * FROM t LIMIT 10 OFFSET 5");
    expect(desc.limit).toBe(10);
    expect(desc.offset).toBe(5);
  });

  it("compiles DISTINCT", () => {
    const desc = sql("SELECT DISTINCT name FROM users");
    expect(desc.distinct).toEqual(["name"]);
  });

  it("compiles aggregate COUNT(*)", () => {
    const desc = sql("SELECT COUNT(*) FROM orders");
    expect(desc.aggregates).toEqual([{ fn: "count", column: "*" }]);
  });

  it("compiles aggregate SUM with alias", () => {
    const desc = sql("SELECT SUM(amount) AS total FROM orders");
    expect(desc.aggregates).toEqual([{ fn: "sum", column: "amount", alias: "total" }]);
  });

  it("compiles COUNT(DISTINCT col) as count_distinct", () => {
    const desc = sql("SELECT COUNT(DISTINCT user_id) FROM orders");
    expect(desc.aggregates).toEqual([{ fn: "count_distinct", column: "user_id" }]);
  });

  it("compiles SUM(expression) with deterministic column key", () => {
    const desc = sql("SELECT SUM(price * quantity) FROM orders");
    expect(desc.aggregates).toEqual([{ fn: "sum", column: "multiply(price,quantity)" }]);
  });

  it("COUNT(expr) does not collide with COUNT(*)", () => {
    const desc = sql("SELECT COUNT(*), COUNT(a + b) FROM t");
    expect(desc.aggregates).toHaveLength(2);
    expect(desc.aggregates![0].column).toBe("*");
    expect(desc.aggregates![1].column).toBe("add(a,b)");
    expect(desc.aggregates![0].column).not.toBe(desc.aggregates![1].column);
  });

  it("compiles GROUP BY", () => {
    const desc = sql("SELECT region, SUM(sales) FROM data GROUP BY region");
    expect(desc.groupBy).toEqual(["region"]);
    expect(desc.aggregates).toEqual([{ fn: "sum", column: "sales" }]);
  });

  it("compiles NEAR vector search", () => {
    const desc = sql("SELECT * FROM items WHERE embedding NEAR [0.1, 0.2, 0.3] TOPK 5");
    expect(desc.vectorSearch).toBeDefined();
    expect(desc.vectorSearch!.column).toBe("embedding");
    expect(desc.vectorSearch!.topK).toBe(5);
    expect(Array.from(desc.vectorSearch!.queryVector)).toEqual([
      expect.closeTo(0.1), expect.closeTo(0.2), expect.closeTo(0.3),
    ]);
  });

  it("compiles JOIN", () => {
    const desc = sql("SELECT * FROM orders JOIN users ON user_id = id");
    expect(desc.join).toBeDefined();
    expect(desc.join!.right.table).toBe("users");
    expect(desc.join!.leftKey).toBe("user_id");
    expect(desc.join!.rightKey).toBe("id");
    expect(desc.join!.type).toBe("inner");
  });

  it("compiles LEFT JOIN", () => {
    const desc = sql("SELECT * FROM orders LEFT JOIN users ON uid = id");
    expect(desc.join!.type).toBe("left");
  });

  it("compiles UNION ALL", () => {
    const desc = sql("SELECT a FROM t1 UNION ALL SELECT b FROM t2");
    expect(desc.setOperation).toBeDefined();
    expect(desc.setOperation!.mode).toBe("union_all");
    expect(desc.setOperation!.right.table).toBe("t2");
  });

  it("compiles UNION (distinct) as union", () => {
    const desc = sql("SELECT a FROM t1 UNION SELECT b FROM t2");
    expect(desc.setOperation!.mode).toBe("union");
  });

  it("compiles INTERSECT", () => {
    const desc = sql("SELECT id FROM t1 INTERSECT SELECT id FROM t2");
    expect(desc.setOperation!.mode).toBe("intersect");
  });

  it("compiles reversed comparison (value op column)", () => {
    const desc = sql("SELECT * FROM t WHERE 10 < age");
    expect(desc.filters).toEqual([{ column: "age", op: "gt", value: 10 }]);
  });

  it("compiles window function", () => {
    const desc = sql("SELECT ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn FROM emp");
    expect(desc.windows).toHaveLength(1);
    expect(desc.windows![0].fn).toBe("row_number");
    expect(desc.windows![0].partitionBy).toEqual(["dept"]);
    expect(desc.windows![0].orderBy).toEqual([{ column: "salary", direction: "desc" }]);
    expect(desc.windows![0].alias).toBe("rn");
  });

  it("compiles OR into filterGroups", () => {
    const desc = sql("SELECT * FROM t WHERE a = 1 OR b = 2");
    expect(desc.filters).toEqual([]);
    expect(desc.filterGroups).toEqual([
      [{ column: "a", op: "eq", value: 1 }],
      [{ column: "b", op: "eq", value: 2 }],
    ]);
  });

  it("handles AND with NEAR (mixed filters)", () => {
    const desc = sql("SELECT * FROM t WHERE status = 'active' AND embedding NEAR [0.1, 0.2] TOPK 3");
    expect(desc.filters).toEqual([{ column: "status", op: "eq", value: "active" }]);
    expect(desc.vectorSearch).toBeDefined();
    expect(desc.vectorSearch!.topK).toBe(3);
  });
});

describe("SQL Compiler - compileFull", () => {
  it("flattens OR into filterGroups", () => {
    const result = sqlFull("SELECT * FROM t WHERE a = 1 OR b = 2");
    expect(result.whereExpr).toBeUndefined();
    expect(result.descriptor.filterGroups).toEqual([
      [{ column: "a", op: "eq", value: 1 }],
      [{ column: "b", op: "eq", value: 2 }],
    ]);
  });

  it("flattens complex OR with AND branches", () => {
    const result = sqlFull("SELECT * FROM t WHERE (a = 1 AND b = 2) OR (c = 3 AND d = 4)");
    expect(result.whereExpr).toBeUndefined();
    expect(result.descriptor.filterGroups).toEqual([
      [{ column: "a", op: "eq", value: 1 }, { column: "b", op: "eq", value: 2 }],
      [{ column: "c", op: "eq", value: 3 }, { column: "d", op: "eq", value: 4 }],
    ]);
  });

  it("returns no whereExpr for simple AND filters", () => {
    const result = sqlFull("SELECT * FROM t WHERE a = 1 AND b = 2");
    expect(result.whereExpr).toBeUndefined();
    expect(result.descriptor.filters).toHaveLength(2);
  });

  it("flattens LIKE into FilterOp", () => {
    const result = sqlFull("SELECT * FROM t WHERE name LIKE '%alice%'");
    expect(result.whereExpr).toBeUndefined();
    expect(result.descriptor.filters).toEqual([{ column: "name", op: "like", value: "%alice%" }]);
  });

  it("flattens NOT LIKE into FilterOp", () => {
    const result = sqlFull("SELECT * FROM t WHERE name NOT LIKE '%test%'");
    expect(result.whereExpr).toBeUndefined();
    expect(result.descriptor.filters).toEqual([{ column: "name", op: "not_like", value: "%test%" }]);
  });

  it("flattens NOT IN into FilterOp", () => {
    const result = sqlFull("SELECT * FROM t WHERE id NOT IN (1, 2, 3)");
    expect(result.whereExpr).toBeUndefined();
    expect(result.descriptor.filters).toEqual([{ column: "id", op: "not_in", value: [1, 2, 3] }]);
  });

  it("flattens NOT BETWEEN into FilterOp", () => {
    const result = sqlFull("SELECT * FROM t WHERE age NOT BETWEEN 18 AND 65");
    expect(result.whereExpr).toBeUndefined();
    expect(result.descriptor.filters).toEqual([{ column: "age", op: "not_between", value: [18, 65] }]);
  });

  it("returns havingExpr for HAVING clause", () => {
    const result = sqlFull("SELECT dept, COUNT(*) FROM t GROUP BY dept HAVING COUNT(*) > 5");
    expect(result.havingExpr).toBeDefined();
    // Aggregate calls should be rewritten to column refs
    expect(result.havingExpr!.kind).toBe("binary");
    if (result.havingExpr!.kind === "binary") {
      expect(result.havingExpr!.left.kind).toBe("column");
      if (result.havingExpr!.left.kind === "column") {
        expect(result.havingExpr!.left.name).toBe("count_*");
      }
    }
  });

  it("HAVING with expression-arg aggregate uses matching column key", () => {
    const result = sqlFull("SELECT dept, SUM(price * qty) FROM t GROUP BY dept HAVING SUM(price * qty) > 100");
    expect(result.havingExpr).toBeDefined();
    if (result.havingExpr!.kind === "binary" && result.havingExpr!.left.kind === "column") {
      expect(result.havingExpr!.left.name).toBe("sum_multiply(price,qty)");
    }
    // Must match the aggregate's column key
    expect(result.descriptor.aggregates![0].column).toBe("multiply(price,qty)");
  });

  it("returns allOrderBy for multi-column ORDER BY", () => {
    const result = sqlFull("SELECT * FROM t ORDER BY name ASC, age DESC");
    expect(result.allOrderBy).toBeDefined();
    expect(result.allOrderBy).toHaveLength(2);
    expect(result.allOrderBy![0]).toEqual({ column: "name", direction: "asc" });
    expect(result.allOrderBy![1]).toEqual({ column: "age", direction: "desc" });
    // Single-column sort fields should not be set when multi-column
    expect(result.descriptor.sortColumn).toBeUndefined();
  });

  it("does not set allOrderBy for single-column ORDER BY", () => {
    const result = sqlFull("SELECT * FROM t ORDER BY name ASC");
    expect(result.allOrderBy).toBeUndefined();
    expect(result.descriptor.sortColumn).toBe("name");
  });

  it("returns computedExprs for CASE in SELECT", () => {
    const result = sqlFull("SELECT CASE WHEN age > 18 THEN 'adult' ELSE 'minor' END AS category FROM t");
    expect(result.computedExprs).toBeDefined();
    expect(result.computedExprs).toHaveLength(1);
    expect(result.computedExprs![0].alias).toBe("category");
    expect(result.computedExprs![0].expr.kind).toBe("case_expr");
  });

  it("returns computedExprs for CAST in SELECT", () => {
    const result = sqlFull("SELECT CAST(age AS float) AS age_f FROM t");
    expect(result.computedExprs).toBeDefined();
    expect(result.computedExprs![0].alias).toBe("age_f");
    expect(result.computedExprs![0].expr.kind).toBe("cast");
  });

  it("returns computedExprs for arithmetic in SELECT", () => {
    const result = sqlFull("SELECT price * quantity AS total FROM t");
    expect(result.computedExprs).toBeDefined();
    expect(result.computedExprs![0].alias).toBe("total");
    expect(result.computedExprs![0].expr.kind).toBe("binary");
  });

  it("CTE inlines simple filters", () => {
    const result = sqlFull("WITH active AS (SELECT * FROM users WHERE status = 'active') SELECT * FROM active WHERE age > 25");
    expect(result.descriptor.table).toBe("users");
    expect(result.descriptor.filters).toHaveLength(2);
    expect(result.descriptor.filters[0]).toMatchObject({ column: "status", op: "eq", value: "active" });
    expect(result.descriptor.filters[1]).toMatchObject({ column: "age", op: "gt", value: 25 });
  });

  it("CTE with un-pushable WHERE propagates whereExpr", () => {
    // UPPER(name) can't be pushed to FilterOp[], so CTE has a whereExpr
    const result = sqlFull("WITH x AS (SELECT * FROM t WHERE UPPER(name) = 'JOHN') SELECT * FROM x");
    expect(result.descriptor.table).toBe("t");
    // The un-pushable predicate should appear as whereExpr, not silently dropped
    expect(result.whereExpr).toBeDefined();
  });

  it("CTE with aggregation is not inlined", () => {
    const result = sqlFull("WITH counts AS (SELECT dept, COUNT(*) AS cnt FROM t GROUP BY dept) SELECT * FROM counts");
    // CTE has groupBy — can't be inlined, table stays as CTE name
    expect(result.descriptor.table).toBe("counts");
  });

  it("CTE with LIMIT is not inlined", () => {
    const result = sqlFull("WITH top5 AS (SELECT * FROM t ORDER BY score DESC LIMIT 5) SELECT * FROM top5");
    // CTE has sort+limit — can't be inlined
    expect(result.descriptor.table).toBe("top5");
  });
});
