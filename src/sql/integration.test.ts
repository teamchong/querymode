import { describe, it, expect } from "vitest";
import { QueryMode } from "../local.js";
import { buildSqlDataFrame } from "./index.js";
import { MaterializedExecutor } from "../client.js";

const SAMPLE_DATA = [
  { id: 1, name: "Alice", age: 30, dept: "eng", salary: 120000 },
  { id: 2, name: "Bob", age: 25, dept: "eng", salary: 95000 },
  { id: 3, name: "Charlie", age: 35, dept: "sales", salary: 110000 },
  { id: 4, name: "Diana", age: 28, dept: "sales", salary: 105000 },
  { id: 5, name: "Eve", age: 32, dept: "eng", salary: 130000 },
];

function makeExecutor() {
  return new MaterializedExecutor({
    rows: SAMPLE_DATA,
    columns: ["id", "name", "age", "dept", "salary"],
    rowCount: SAMPLE_DATA.length,
    totalRows: SAMPLE_DATA.length,
    scannedBytes: 0,
    elapsedMs: 0,
  });
}

describe("SQL Integration", () => {
  it("SELECT * returns all rows", async () => {
    const result = await QueryMode.fromJSON(SAMPLE_DATA, "data")
      .filter("age", "gt", 0)
      .collect();
    expect(result.rowCount).toBe(5);
  });

  it("filters with WHERE equality", async () => {
    const result = await QueryMode.fromJSON(SAMPLE_DATA, "data")
      .filter("dept", "eq", "eng")
      .collect();
    expect(result.rowCount).toBe(3);
    expect(result.rows.every(r => r.dept === "eng")).toBe(true);
  });

  it("filters with WHERE comparison", async () => {
    const result = await QueryMode.fromJSON(SAMPLE_DATA, "data")
      .filter("age", "gt", 30)
      .collect();
    expect(result.rowCount).toBe(2);
  });

  it("projects specific columns", async () => {
    const result = await QueryMode.fromJSON(SAMPLE_DATA, "data")
      .select("name", "age")
      .collect();
    expect(result.columns).toEqual(["name", "age"]);
    expect(Object.keys(result.rows[0])).toEqual(["name", "age"]);
  });

  it("sorts results", async () => {
    const result = await QueryMode.fromJSON(SAMPLE_DATA, "data")
      .sort("age", "desc")
      .collect();
    expect(result.rows[0].name).toBe("Charlie");
    expect(result.rows[result.rows.length - 1].name).toBe("Bob");
  });

  it("limits results", async () => {
    const result = await QueryMode.fromJSON(SAMPLE_DATA, "data")
      .limit(2)
      .collect();
    expect(result.rowCount).toBe(2);
  });

  it("offsets results", async () => {
    const result = await QueryMode.fromJSON(SAMPLE_DATA, "data")
      .offset(3)
      .collect();
    expect(result.rowCount).toBe(2);
  });

  it("composes filter + sort + limit via DataFrame API", async () => {
    const result = await QueryMode.fromJSON(SAMPLE_DATA, "data")
      .filter("dept", "eq", "eng")
      .sort("salary", "desc")
      .limit(2)
      .collect();
    expect(result.rowCount).toBe(2);
    expect(result.rows[0].name).toBe("Eve");
    expect(result.rows[1].name).toBe("Alice");
  });

  it("sqlToDescriptor compiles aggregation COUNT(*)", async () => {
    const { sqlToDescriptor } = await import("./index.js");
    const desc = sqlToDescriptor("SELECT COUNT(*) FROM orders");
    expect(desc.aggregates).toEqual([{ fn: "count", column: "*" }]);
  });

  it("sqlToDescriptor compiles SUM with GROUP BY", async () => {
    const { sqlToDescriptor } = await import("./index.js");
    const desc = sqlToDescriptor("SELECT dept, SUM(salary) FROM employees GROUP BY dept");
    expect(desc.groupBy).toEqual(["dept"]);
    expect(desc.aggregates).toEqual([{ fn: "sum", column: "salary" }]);
  });

  it("sqlToDescriptor produces correct table name", async () => {
    const { sqlToDescriptor } = await import("./index.js");
    const desc = sqlToDescriptor("SELECT name, age FROM employees WHERE age > 25 ORDER BY name LIMIT 10");
    expect(desc.table).toBe("employees");
    expect(desc.projections).toEqual(["name", "age"]);
    expect(desc.filters).toEqual([{ column: "age", op: "gt", value: 25 }]);
    expect(desc.sortColumn).toBe("name");
    expect(desc.sortDirection).toBe("asc");
    expect(desc.limit).toBe(10);
  });

  it("sqlToDescriptor handles BETWEEN", async () => {
    const { sqlToDescriptor } = await import("./index.js");
    const desc = sqlToDescriptor("SELECT * FROM t WHERE age BETWEEN 20 AND 40");
    expect(desc.filters).toEqual([
      { column: "age", op: "between", value: [20, 40] },
    ]);
  });
});

describe("SQL Integration - buildSqlDataFrame", () => {
  it("handles OR conditions via wrapper", async () => {
    const df = buildSqlDataFrame("SELECT * FROM data WHERE dept = 'eng' OR age > 30", makeExecutor());
    const result = await df.collect();
    // eng: Alice(30), Bob(25), Eve(32) + age>30: Charlie(35) = 4 unique
    const names = result.rows.map(r => r.name).sort();
    expect(names).toEqual(["Alice", "Bob", "Charlie", "Eve"]);
  });

  it("handles LIKE via wrapper", async () => {
    const df = buildSqlDataFrame("SELECT * FROM data WHERE name LIKE '%li%'", makeExecutor());
    const result = await df.collect();
    // Alice and Charlie contain "li"
    expect(result.rowCount).toBe(2);
    const names = result.rows.map(r => r.name).sort();
    expect(names).toEqual(["Alice", "Charlie"]);
  });

  it("handles NOT IN via wrapper", async () => {
    const df = buildSqlDataFrame("SELECT * FROM data WHERE dept NOT IN ('sales')", makeExecutor());
    const result = await df.collect();
    expect(result.rowCount).toBe(3);
    expect(result.rows.every(r => r.dept === "eng")).toBe(true);
  });

  it("handles multi-column ORDER BY via wrapper", async () => {
    const df = buildSqlDataFrame("SELECT * FROM data ORDER BY dept ASC, salary DESC", makeExecutor());
    const result = await df.collect();
    // eng sorted by salary desc: Eve(130k), Alice(120k), Bob(95k)
    // sales sorted by salary desc: Charlie(110k), Diana(105k)
    expect(result.rows[0].name).toBe("Eve");
    expect(result.rows[1].name).toBe("Alice");
    expect(result.rows[2].name).toBe("Bob");
    expect(result.rows[3].name).toBe("Charlie");
    expect(result.rows[4].name).toBe("Diana");
  });

  it("handles multi-column ORDER BY with LIMIT", async () => {
    const df = buildSqlDataFrame("SELECT * FROM data ORDER BY dept ASC, salary DESC LIMIT 3", makeExecutor());
    const result = await df.collect();
    expect(result.rowCount).toBe(3);
    expect(result.rows[0].name).toBe("Eve");
    expect(result.rows[1].name).toBe("Alice");
    expect(result.rows[2].name).toBe("Bob");
  });

  it("handles CASE expression in SELECT via wrapper", async () => {
    const df = buildSqlDataFrame(
      "SELECT name, CASE WHEN age > 30 THEN 'senior' ELSE 'junior' END AS level FROM data",
      makeExecutor(),
    );
    const result = await df.collect();
    const alice = result.rows.find(r => r.name === "Alice");
    const charlie = result.rows.find(r => r.name === "Charlie");
    const bob = result.rows.find(r => r.name === "Bob");
    expect(alice?.level).toBe("junior"); // age=30, not >30
    expect(charlie?.level).toBe("senior"); // age=35
    expect(bob?.level).toBe("junior"); // age=25
  });

  it("handles CAST in SELECT via wrapper", async () => {
    const df = buildSqlDataFrame(
      "SELECT name, CAST(age AS text) AS age_str FROM data",
      makeExecutor(),
    );
    const result = await df.collect();
    const alice = result.rows.find(r => r.name === "Alice");
    expect(alice?.age_str).toBe("30");
  });

  it("handles arithmetic in SELECT via wrapper", async () => {
    const df = buildSqlDataFrame(
      "SELECT name, salary / 1000 AS salary_k FROM data",
      makeExecutor(),
    );
    const result = await df.collect();
    const alice = result.rows.find(r => r.name === "Alice");
    expect(alice?.salary_k).toBe(120);
  });

  it("passes through simple AND filters without wrapper", async () => {
    const df = buildSqlDataFrame("SELECT * FROM data WHERE age > 25 AND dept = 'eng'", makeExecutor());
    const result = await df.collect();
    // eng + age>25: Alice(30), Eve(32)
    expect(result.rowCount).toBe(2);
  });
});
