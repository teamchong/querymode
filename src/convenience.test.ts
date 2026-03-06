import { describe, it, expect } from "vitest";
import { createFromJSON, createFromCSV, createDemo } from "./convenience.js";
import { QueryMode } from "./local.js";
import { QueryModeError } from "./errors.js";

describe("createFromJSON", () => {
  it("creates DataFrame from typical array", async () => {
    const data = [
      { id: 1, name: "Alice", age: 30 },
      { id: 2, name: "Bob", age: 25 },
      { id: 3, name: "Charlie", age: 35 },
    ];
    const df = createFromJSON(data);
    const result = await df.collect();
    expect(result.rowCount).toBe(3);
    expect(result.columns).toEqual(["id", "name", "age"]);
    expect(result.rows[0]).toEqual({ id: 1, name: "Alice", age: 30 });
  });

  it("handles empty array", async () => {
    const df = createFromJSON([]);
    const result = await df.collect();
    expect(result.rowCount).toBe(0);
    expect(result.rows).toEqual([]);
  });

  it("handles null values", async () => {
    const data = [
      { id: 1, name: "Alice", score: null },
      { id: 2, name: null, score: 95 },
    ];
    const df = createFromJSON(data);
    const result = await df.collect();
    expect(result.rowCount).toBe(2);
    expect(result.rows[0].score).toBe(null);
    expect(result.rows[1].name).toBe(null);
  });
});

describe("createFromCSV", () => {
  it("parses comma-delimited CSV", async () => {
    const csv = "id,name,age\n1,Alice,30\n2,Bob,25\n3,Charlie,35";
    const df = await createFromCSV(csv);
    const result = await df.collect();
    expect(result.rowCount).toBe(3);
    expect(result.columns).toEqual(["id", "name", "age"]);
  });

  it("parses tab-delimited CSV", async () => {
    const csv = "id\tname\tage\n1\tAlice\t30\n2\tBob\t25";
    const df = await createFromCSV(csv);
    const result = await df.collect();
    expect(result.rowCount).toBe(2);
    expect(result.columns).toEqual(["id", "name", "age"]);
  });
});

describe("createDemo", () => {
  it("returns 1000 rows", async () => {
    const df = createDemo();
    const result = await df.collect();
    expect(result.rowCount).toBe(1000);
    expect(result.columns).toEqual(["id", "region", "category", "amount", "created_at"]);
  });

  it("is deterministic across calls", async () => {
    const df1 = createDemo();
    const df2 = createDemo();
    const r1 = await df1.collect();
    const r2 = await df2.collect();
    expect(r1.rows).toEqual(r2.rows);
  });

  it("has expected column types", async () => {
    const df = createDemo();
    const result = await df.collect();
    const row = result.rows[0];
    expect(typeof row.id).toBe("number");
    expect(typeof row.region).toBe("string");
    expect(typeof row.category).toBe("string");
    expect(typeof row.amount).toBe("number");
    expect(typeof row.created_at).toBe("string");
  });
});

describe("QueryMode static factories", () => {
  it("fromJSON is accessible on QueryMode", async () => {
    const df = QueryMode.fromJSON([{ x: 1 }]);
    const result = await df.collect();
    expect(result.rowCount).toBe(1);
  });

  it("demo is accessible on QueryMode", async () => {
    const df = QueryMode.demo();
    const result = await df.collect();
    expect(result.rowCount).toBe(1000);
  });

  it("supports chaining: filter → sort → limit → collect", async () => {
    const data = [
      { id: 1, name: "Alice", age: 30 },
      { id: 2, name: "Bob", age: 25 },
      { id: 3, name: "Charlie", age: 35 },
      { id: 4, name: "Diana", age: 28 },
    ];
    const result = await QueryMode.fromJSON(data)
      .filter("age", "gt", 26)
      .sort("age", "desc")
      .limit(2)
      .collect();
    expect(result.rowCount).toBe(2);
    expect(result.rows[0].name).toBe("Charlie");
    expect(result.rows[1].name).toBe("Alice");
  });
});

describe("error messages", () => {
  it("TABLE_NOT_FOUND includes helpful suggestions", () => {
    const err = QueryModeError.from(
      Object.assign(new Error("ENOENT"), { code: "ENOENT", path: "/missing/file" }),
      { table: "missing_table" },
    );
    expect(err.code).toBe("TABLE_NOT_FOUND");
    expect(err.message).toContain("fromJSON()");
    expect(err.message).toContain("fromCSV()");
  });

  it("INVALID_FORMAT lists supported formats", () => {
    const err = QueryModeError.from(new Error("Invalid file format"), { table: "bad.xyz" });
    expect(err.code).toBe("INVALID_FORMAT");
    expect(err.message).toContain(".parquet");
    expect(err.message).toContain(".csv");
    expect(err.message).toContain(".json");
  });
});
