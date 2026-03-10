import { describe, it, expect } from "vitest";
import { createFromJSON } from "./convenience.js";
import { QueryMode } from "./local.js";

describe("MaterializedExecutor", () => {
  const data = [
    { id: 1, name: "Alice", region: "us", amount: 100 },
    { id: 2, name: "Bob", region: "eu", amount: 200 },
    { id: 3, name: "Charlie", region: "us", amount: 150 },
    { id: 4, name: "Diana", region: "asia", amount: 300 },
    { id: 5, name: "Eve", region: "eu", amount: 50 },
  ];

  describe("filters", () => {
    it("eq filter", async () => {
      const result = await createFromJSON(data).filter("region", "eq", "us").collect();
      expect(result.rowCount).toBe(2);
      expect(result.rows.map(r => r.name)).toEqual(["Alice", "Charlie"]);
    });

    it("neq filter", async () => {
      const result = await createFromJSON(data).filter("region", "neq", "us").collect();
      expect(result.rowCount).toBe(3);
    });

    it("gt filter", async () => {
      const result = await createFromJSON(data).filter("amount", "gt", 150).collect();
      expect(result.rowCount).toBe(2);
      expect(result.rows.map(r => r.name)).toEqual(["Bob", "Diana"]);
    });

    it("gte filter", async () => {
      const result = await createFromJSON(data).filter("amount", "gte", 150).collect();
      expect(result.rowCount).toBe(3);
    });

    it("lt filter", async () => {
      const result = await createFromJSON(data).filter("amount", "lt", 150).collect();
      expect(result.rowCount).toBe(2);
    });

    it("lte filter", async () => {
      const result = await createFromJSON(data).filter("amount", "lte", 150).collect();
      expect(result.rowCount).toBe(3);
    });

    it("in filter", async () => {
      const result = await createFromJSON(data).filter("region", "in", ["us", "asia"]).collect();
      expect(result.rowCount).toBe(3);
    });

    it("not_in filter", async () => {
      const result = await createFromJSON(data).filter("region", "not_in", ["us", "asia"]).collect();
      expect(result.rowCount).toBe(2);
    });

    it("between filter", async () => {
      const result = await createFromJSON(data).filter("amount", "between", [100, 200]).collect();
      expect(result.rowCount).toBe(3); // 100, 200, 150
    });

    it("is_null filter", async () => {
      const dataWithNull = [
        { id: 1, score: 90 },
        { id: 2, score: null },
        { id: 3, score: 80 },
      ];
      const result = await createFromJSON(dataWithNull).whereNull("score").collect();
      expect(result.rowCount).toBe(1);
      expect(result.rows[0].id).toBe(2);
    });

    it("is_not_null filter", async () => {
      const dataWithNull = [
        { id: 1, score: 90 },
        { id: 2, score: null },
        { id: 3, score: 80 },
      ];
      const result = await createFromJSON(dataWithNull).whereNotNull("score").collect();
      expect(result.rowCount).toBe(2);
    });

    it("multiple AND filters", async () => {
      const result = await createFromJSON(data)
        .filter("region", "eq", "us")
        .filter("amount", "gt", 100)
        .collect();
      expect(result.rowCount).toBe(1);
      expect(result.rows[0].name).toBe("Charlie");
    });
  });

  describe("OR filter groups", () => {
    it("matches rows in any group", async () => {
      const result = await createFromJSON(data)
        .whereOr(
          [{ column: "region", op: "eq", value: "us" }],
          [{ column: "region", op: "eq", value: "asia" }],
        )
        .collect();
      expect(result.rowCount).toBe(3);
    });
  });

  describe("projections", () => {
    it("select keeps only specified columns", async () => {
      const result = await createFromJSON(data).select("name", "region").collect();
      expect(result.columns).toEqual(["name", "region"]);
      expect(Object.keys(result.rows[0])).toEqual(["name", "region"]);
    });
  });

  describe("sort", () => {
    it("sorts ascending", async () => {
      const result = await createFromJSON(data).sort("amount", "asc").collect();
      const amounts = result.rows.map(r => r.amount);
      expect(amounts).toEqual([50, 100, 150, 200, 300]);
    });

    it("sorts descending", async () => {
      const result = await createFromJSON(data).sort("amount", "desc").collect();
      const amounts = result.rows.map(r => r.amount);
      expect(amounts).toEqual([300, 200, 150, 100, 50]);
    });

    it("sorts strings", async () => {
      const result = await createFromJSON(data).sort("name", "asc").collect();
      expect(result.rows[0].name).toBe("Alice");
      expect(result.rows[4].name).toBe("Eve");
    });

    it("pushes nulls to end", async () => {
      const dataWithNull = [
        { id: 1, val: null },
        { id: 2, val: 10 },
        { id: 3, val: 5 },
      ];
      const result = await createFromJSON(dataWithNull).sort("val", "asc").collect();
      expect(result.rows[2].val).toBe(null);
    });
  });

  describe("limit and offset", () => {
    it("limit restricts row count", async () => {
      const result = await createFromJSON(data).limit(2).collect();
      expect(result.rowCount).toBe(2);
    });

    it("offset skips rows", async () => {
      const result = await createFromJSON(data).offset(3).collect();
      expect(result.rowCount).toBe(2);
    });

    it("limit + offset for pagination", async () => {
      const result = await createFromJSON(data).sort("id", "asc").offset(1).limit(2).collect();
      expect(result.rowCount).toBe(2);
      expect(result.rows[0].name).toBe("Bob");
      expect(result.rows[1].name).toBe("Charlie");
    });
  });

  describe("aggregation", () => {
    it("sum aggregation", async () => {
      const result = await createFromJSON(data)
        .aggregate("sum", "amount", "total")
        .collect();
      expect(result.rows[0].total).toBe(800);
    });

    it("avg aggregation", async () => {
      const result = await createFromJSON(data)
        .aggregate("avg", "amount", "avg_amount")
        .collect();
      expect(result.rows[0].avg_amount).toBe(160);
    });

    it("min aggregation", async () => {
      const result = await createFromJSON(data)
        .aggregate("min", "amount", "min_amount")
        .collect();
      expect(result.rows[0].min_amount).toBe(50);
    });

    it("max aggregation", async () => {
      const result = await createFromJSON(data)
        .aggregate("max", "amount", "max_amount")
        .collect();
      expect(result.rows[0].max_amount).toBe(300);
    });

    it("count aggregation", async () => {
      const result = await createFromJSON(data)
        .aggregate("count", "id", "cnt")
        .collect();
      expect(result.rows[0].cnt).toBe(5);
    });

    it("count with nulls skips null values", async () => {
      const dataWithNull = [
        { id: 1, score: 90 },
        { id: 2, score: null },
        { id: 3, score: 80 },
      ];
      const result = await createFromJSON(dataWithNull)
        .aggregate("count", "score", "cnt")
        .collect();
      expect(result.rows[0].cnt).toBe(2);
    });

    it("count(*) counts all rows including nulls", async () => {
      const dataWithNull = [
        { id: 1, score: 90 },
        { id: 2, score: null },
        { id: 3, score: 80 },
      ];
      const result = await createFromJSON(dataWithNull)
        .aggregate("count", "*", "cnt")
        .collect();
      expect(result.rows[0].cnt).toBe(3);
    });

    it("groupBy + aggregate", async () => {
      const result = await createFromJSON(data)
        .groupBy("region")
        .aggregate("sum", "amount", "total")
        .aggregate("count", "id", "cnt")
        .sort("total", "desc")
        .collect();
      expect(result.rowCount).toBe(3);
      // asia=300, us=250, eu=250
      const asia = result.rows.find(r => r.region === "asia");
      expect(asia?.total).toBe(300);
      expect(asia?.cnt).toBe(1);
      const us = result.rows.find(r => r.region === "us");
      expect(us?.total).toBe(250);
      expect(us?.cnt).toBe(2);
    });

    it("aggregation with bigint values", async () => {
      const bigData = [
        { id: 1n, amount: 100n },
        { id: 2n, amount: 200n },
        { id: 3n, amount: 300n },
      ];
      const result = await createFromJSON(bigData)
        .aggregate("sum", "amount", "total")
        .collect();
      // BigInt values should be converted to Number for aggregation
      expect(result.rows[0].total).toBe(600);
    });

    it("aggregation on empty data returns null", async () => {
      const result = await createFromJSON([] as Record<string, unknown>[])
        .aggregate("sum", "amount", "total")
        .collect();
      // SQL standard: SELECT SUM(x) FROM empty_table returns 1 row with null
      expect(result.rowCount).toBe(1);
      expect(result.rows[0].total).toBe(null);
    });
  });

  describe("computed columns", () => {
    it("adds computed column", async () => {
      const result = await createFromJSON(data)
        .computed("double_amount", (row) => (row.amount as number) * 2)
        .collect();
      expect(result.rows[0].double_amount).toBe(200);
      expect(result.rows[3].double_amount).toBe(600);
    });

    it("computed column available to subsequent filter", async () => {
      const result = await createFromJSON(data)
        .computed("double_amount", (row) => (row.amount as number) * 2)
        .filter("double_amount", "gt", 400)
        .collect();
      expect(result.rowCount).toBe(1);
      expect(result.rows[0].name).toBe("Diana");
    });
  });

  describe("chained operations", () => {
    it("filter → sort → limit → select", async () => {
      const result = await createFromJSON(data)
        .filter("region", "neq", "asia")
        .sort("amount", "desc")
        .limit(2)
        .select("name", "amount")
        .collect();
      expect(result.rowCount).toBe(2);
      expect(result.rows[0]).toEqual({ name: "Bob", amount: 200 });
      expect(result.rows[1]).toEqual({ name: "Charlie", amount: 150 });
    });

    it("groupBy → aggregate → sort → limit", async () => {
      const result = await createFromJSON(data)
        .groupBy("region")
        .aggregate("sum", "amount", "total")
        .sort("total", "desc")
        .limit(2)
        .collect();
      expect(result.rowCount).toBe(2);
      expect(result.rows[0].region).toBe("asia");
    });
  });

  describe("terminal methods", () => {
    it("first() returns first row", async () => {
      const row = await createFromJSON(data).sort("id", "asc").first();
      expect(row?.name).toBe("Alice");
    });

    it("first() returns null for empty result", async () => {
      const row = await createFromJSON(data).filter("region", "eq", "mars").first();
      expect(row).toBeNull();
    });

    it("count() returns row count", async () => {
      const count = await createFromJSON(data).filter("region", "eq", "us").count();
      expect(count).toBe(2);
    });

    it("exists() returns true when rows match", async () => {
      const exists = await createFromJSON(data).filter("region", "eq", "us").exists();
      expect(exists).toBe(true);
    });

    it("exists() returns false when no rows match", async () => {
      const exists = await createFromJSON(data).filter("region", "eq", "mars").exists();
      expect(exists).toBe(false);
    });

    it("head(n) returns first n rows", async () => {
      const rows = await createFromJSON(data).sort("id", "asc").head(3);
      expect(rows.length).toBe(3);
      expect(rows[0].name).toBe("Alice");
    });

    it("explain() returns plan info", async () => {
      const plan = await createFromJSON(data).explain();
      expect(plan.totalRows).toBe(5);
      expect(plan.metaCached).toBe(true);
    });

    it("describe() returns schema", async () => {
      const schema = await createFromJSON(data).describe();
      expect(schema.totalRows).toBe(5);
      expect(schema.columns.length).toBe(4);
    });
  });

  describe("convenience methods", () => {
    it("shape() returns rows and columns", async () => {
      const s = await createFromJSON(data).shape();
      expect(s.rows).toBe(5);
      expect(s.columns).toBe(4);
    });

    it("dtypes() returns type map", async () => {
      const dt = await createFromJSON(data).dtypes();
      expect(Object.keys(dt).length).toBe(4);
    });

    it("fillNull() replaces null values", async () => {
      const dataWithNull = [
        { id: 1, score: null },
        { id: 2, score: 90 },
      ];
      const result = await createFromJSON(dataWithNull).fillNull("score", 0).collect();
      expect(result.rows[0].score).toBe(0);
      expect(result.rows[1].score).toBe(90);
    });

    it("cast() to string", async () => {
      const result = await createFromJSON(data).cast("amount", "string").collect();
      expect(typeof result.rows[0].amount).toBe("string");
    });

    it("cast() to number", async () => {
      const strData = [{ val: "42" }, { val: "99" }];
      const result = await createFromJSON(strData).cast("val", "number").collect();
      expect(result.rows[0].val).toBe(42);
    });

    it("sample() returns correct count", async () => {
      const bigData = Array.from({ length: 100 }, (_, i) => ({ id: i }));
      const samples = await createFromJSON(bigData).sample(10);
      expect(samples.length).toBe(10);
    });

    it("valueCounts() sorted descending", async () => {
      const vc = await createFromJSON(data).valueCounts("region");
      expect(vc[0].count).toBeGreaterThanOrEqual(vc[vc.length - 1].count);
      // us=2, eu=2, asia=1
      expect(vc.find(v => v.value === "asia")?.count).toBe(1);
    });

    it("toJSON() roundtrips", async () => {
      const json = await createFromJSON(data).toJSON();
      const parsed = JSON.parse(json);
      expect(parsed.length).toBe(5);
      expect(parsed[0].name).toBe("Alice");
    });

    it("toCSV() has correct header and rows", async () => {
      const csv = await createFromJSON(data).select("id", "name").toCSV();
      const lines = csv.split("\n");
      expect(lines[0]).toBe("id,name");
      expect(lines.length).toBe(6); // header + 5 rows
    });
  });

  describe("immutability", () => {
    it("filter returns new DataFrame, original unchanged", async () => {
      const df = createFromJSON(data);
      const filtered = df.filter("region", "eq", "us");
      const allResult = await df.collect();
      const filteredResult = await filtered.collect();
      expect(allResult.rowCount).toBe(5);
      expect(filteredResult.rowCount).toBe(2);
    });

    it("multiple branches from same DataFrame", async () => {
      const df = createFromJSON(data);
      const branch1 = df.filter("region", "eq", "us").count();
      const branch2 = df.filter("region", "eq", "eu").count();
      const [c1, c2] = await Promise.all([branch1, branch2]);
      expect(c1).toBe(2);
      expect(c2).toBe(2);
    });
  });

  describe("schema evolution", () => {
    it("addColumn adds column with default value to all rows", async () => {
      const df = createFromJSON(data);
      const result = await df.addColumn("status", "utf8", "active");
      expect(result.operation).toBe("add_column");
      expect(result.column).toBe("status");
      expect(result.columnsAfter).toContain("status");

      // Verify all rows have the new column
      const rows = await df.collect();
      expect(rows.rows.every(r => r.status === "active")).toBe(true);
    });

    it("addColumn with null default", async () => {
      const df = createFromJSON(data);
      await df.addColumn("notes", "utf8", null);
      const rows = await df.collect();
      expect(rows.rows.every(r => r.notes === null)).toBe(true);
    });

    it("dropColumn removes column from all rows", async () => {
      const df = createFromJSON(data);
      const result = await df.dropColumn("amount");
      expect(result.operation).toBe("drop_column");
      expect(result.column).toBe("amount");
      expect(result.columnsAfter).not.toContain("amount");

      // Verify column is gone
      const rows = await df.collect();
      expect(rows.rows.every(r => !("amount" in r))).toBe(true);
    });

    it("addColumn then query filters on new column", async () => {
      const df = createFromJSON(data);
      await df.addColumn("tier", "utf8", "free");
      const result = await df.filter("tier", "eq", "free").collect();
      expect(result.rowCount).toBe(5);
    });

    it("dropColumn then query excludes dropped column", async () => {
      const df = createFromJSON(data);
      await df.dropColumn("region");
      const result = await df.collect();
      expect(result.columns).not.toContain("region");
      expect(Object.keys(result.rows[0])).not.toContain("region");
    });
  });
});
