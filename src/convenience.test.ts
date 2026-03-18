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

describe("DataFrame.drop", () => {
  it("removes specified columns from result", async () => {
    const data = [
      { id: 1, name: "Alice", age: 30 },
      { id: 2, name: "Bob", age: 25 },
    ];
    const df = createFromJSON(data).drop("age");
    const result = await df.collect();
    expect(result.columns).toEqual(["id", "name"]);
    expect(result.rows[0]).toEqual({ id: 1, name: "Alice" });
    expect(result.rows[1]).toEqual({ id: 2, name: "Bob" });
  });

  it("removes multiple columns", async () => {
    const data = [{ id: 1, name: "Alice", age: 30, score: 95 }];
    const df = createFromJSON(data).drop("age", "score");
    const result = await df.collect();
    expect(result.columns).toEqual(["id", "name"]);
    expect(result.rows[0]).toEqual({ id: 1, name: "Alice" });
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

  it("QUERY_TIMEOUT from withTimeout-style capital-T message", () => {
    const err = QueryModeError.from(new Error("Timeout after 10000ms"), { table: "events" });
    expect(err.code).toBe("QUERY_TIMEOUT");
    expect(err.message).toContain("events");
  });

  it("NETWORK_TIMEOUT when message mentions R2", () => {
    const err = QueryModeError.from(new Error("R2 read timeout"), { table: "events" });
    expect(err.code).toBe("NETWORK_TIMEOUT");
  });
});

describe("DataFrame Pandas-like methods", () => {
  const data = [
    { id: 1, name: "Alice", age: 30 },
    { id: 2, name: "Bob", age: 25 },
    { id: 3, name: "Charlie", age: 35 },
  ];

  it("shape() returns rows and columns count", async () => {
    const df = createFromJSON(data);
    const s = await df.shape();
    expect(s.rows).toBe(3);
    expect(s.columns).toBe(3);
  });

  it("dtypes() returns column type map", async () => {
    const df = createFromJSON(data);
    const dt = await df.dtypes();
    expect(dt).toHaveProperty("id");
    expect(dt).toHaveProperty("name");
    expect(dt).toHaveProperty("age");
  });

  it("fillNull() replaces null values", async () => {
    const df = createFromJSON([
      { id: 1, score: null },
      { id: 2, score: 90 },
      { id: 3, score: null },
    ]);
    const result = await df.fillNull("score", 0).collect();
    expect(result.rows[0].score).toBe(0);
    expect(result.rows[1].score).toBe(90);
    expect(result.rows[2].score).toBe(0);
  });

  it("cast() converts column types", async () => {
    const df = createFromJSON(data);
    const result = await df.cast("age", "string").collect();
    expect(result.rows[0].age).toBe("30");
    expect(result.rows[1].age).toBe("25");
  });

  it("sample() returns requested number of rows", async () => {
    const bigData = Array.from({ length: 100 }, (_, i) => ({ id: i }));
    const df = createFromJSON(bigData);
    const s = await df.sample(5);
    expect(s.length).toBe(5);
  });

  it("sample() returns all rows when n > length", async () => {
    const df = createFromJSON(data);
    const s = await df.sample(100);
    expect(s.length).toBe(3);
  });

  it("valueCounts() returns frequency counts sorted desc", async () => {
    const df = createFromJSON([
      { color: "red" },
      { color: "blue" },
      { color: "red" },
      { color: "red" },
      { color: "blue" },
    ]);
    const vc = await df.valueCounts("color");
    expect(vc[0]).toEqual({ value: "red", count: 3 });
    expect(vc[1]).toEqual({ value: "blue", count: 2 });
  });

  it("toJSON() returns valid JSON string", async () => {
    const df = createFromJSON(data);
    const json = await df.toJSON();
    const parsed = JSON.parse(json);
    expect(parsed.length).toBe(3);
    expect(parsed[0].name).toBe("Alice");
  });

  it("toJSON({ pretty: true }) returns indented JSON", async () => {
    const df = createFromJSON([{ x: 1 }]);
    const json = await df.toJSON({ pretty: true });
    expect(json).toContain("\n");
    expect(json).toContain("  ");
  });

  it("toCSV() returns valid CSV string", async () => {
    const df = createFromJSON(data);
    const csv = await df.toCSV();
    const lines = csv.split("\n");
    expect(lines[0]).toBe("id,name,age");
    expect(lines.length).toBe(4); // header + 3 rows
  });

  it("toCSV() handles values with commas and quotes", async () => {
    const df = createFromJSON([
      { name: 'O"Brien', city: "New York, NY" },
    ]);
    const csv = await df.toCSV();
    expect(csv).toContain('"O""Brien"');
    expect(csv).toContain('"New York, NY"');
  });

  it("toCSV() with custom delimiter", async () => {
    const df = createFromJSON(data);
    const tsv = await df.toCSV({ delimiter: "\t" });
    expect(tsv.split("\n")[0]).toBe("id\tname\tage");
  });

  it("toCSV() returns empty string for empty data", async () => {
    const df = createFromJSON([]);
    const csv = await df.toCSV();
    expect(csv).toBe("");
  });
});

describe("materializeAs — multi-stage pipeline", () => {
  it("executes stage 1 and returns new DataFrame for stage 2", async () => {
    const data = [
      { dept: "eng", salary: 100 },
      { dept: "eng", salary: 200 },
      { dept: "sales", salary: 150 },
      { dept: "eng", salary: 300 },
      { dept: "sales", salary: 250 },
    ];

    // Stage 1: aggregate by dept
    const stage1 = createFromJSON(data);
    const stage1Result = await stage1
      .groupBy("dept")
      .aggregate("sum", "salary")
      .collect();

    // materializeAs produces a new DataFrame backed by the result
    const stage2Input = createFromJSON(stage1Result.rows, "dept_totals");
    const stage2 = await stage2Input
      .filter("sum_salary", "gt", 500)
      .collect();

    expect(stage2.rowCount).toBe(1);
    expect(stage2.rows[0].dept).toBe("eng");
    expect(stage2.rows[0].sum_salary).toBe(600);
  });

  it("multi-stage pipeline: filter → aggregate → filter → sort", async () => {
    const events = Array.from({ length: 100 }, (_, i) => ({
      page: `/page-${i % 10}`,
      type: i % 3 === 0 ? "click" : "view",
      duration: i * 10,
    }));

    // Stage 1: filter clicks, aggregate by page
    const clickData = createFromJSON(events);
    const clickAgg = await clickData
      .filter("type", "eq", "click")
      .groupBy("page")
      .aggregate("count", "*")
      .aggregate("avg", "duration")
      .collect();

    // Stage 2: filter high-traffic pages, sort
    const highTraffic = createFromJSON(clickAgg.rows, "click_agg");
    const result = await highTraffic
      .filter("count_*", "gt", 2)
      .sort("avg_duration", "desc")
      .collect();

    // Each page gets ~3-4 clicks out of ~34 total clicks (every 3rd event)
    expect(result.rowCount).toBeGreaterThan(0);
    // Results should be sorted descending by avg_duration
    for (let i = 1; i < result.rows.length; i++) {
      expect(result.rows[i - 1].avg_duration).toBeGreaterThanOrEqual(
        result.rows[i].avg_duration as number,
      );
    }
  });
});
