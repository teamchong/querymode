import { describe, it, expect } from "vitest";
import { parseCsvFull } from "./csv-reader.js";
import { parseJsonFull } from "./json-reader.js";

// ---------------------------------------------------------------------------
// CSV: full-dataset type inference (no 256-row sample limit)
// ---------------------------------------------------------------------------

describe("csv-reader type inference", () => {
  it("infers int64 from all rows, not just first 256", () => {
    // Build a CSV with 300 rows — all integers
    const header = "id,value";
    const rows = Array.from({ length: 300 }, (_, i) => `${i},${i * 10}`);
    const csv = [header, ...rows].join("\n");
    const parsed = parseCsvFull(csv);
    expect(parsed.types[0]).toBe("int64");
    expect(parsed.types[1]).toBe("int64");
    // Row 299 should be correctly typed as bigint
    expect(parsed.columns[0][299]).toBe(299n);
  });

  it("detects float in late rows (beyond row 256)", () => {
    const header = "val";
    // First 260 rows are integers, row 261 is a float
    const rows: string[] = [];
    for (let i = 0; i < 260; i++) rows.push(`${i}`);
    rows.push("3.14");
    const csv = [header, ...rows].join("\n");
    const parsed = parseCsvFull(csv);
    // Should be float64, not int64 (the float at row 261 widens the type)
    expect(parsed.types[0]).toBe("float64");
    // The float value should be preserved as number, not truncated to bigint
    expect(parsed.columns[0][260]).toBeCloseTo(3.14);
  });

  it("detects string in late rows (beyond row 256)", () => {
    const header = "val";
    const rows: string[] = [];
    for (let i = 0; i < 260; i++) rows.push(`${i}`);
    rows.push("hello");
    const csv = [header, ...rows].join("\n");
    const parsed = parseCsvFull(csv);
    // String in row 261 should widen type to utf8
    expect(parsed.types[0]).toBe("utf8");
    expect(parsed.columns[0][260]).toBe("hello");
  });
});

// ---------------------------------------------------------------------------
// JSON: full-dataset schema inference (no 256-row sample limit)
// ---------------------------------------------------------------------------

describe("json-reader schema inference", () => {
  it("discovers keys appearing after row 256", () => {
    const objects: Record<string, unknown>[] = [];
    for (let i = 0; i < 260; i++) {
      objects.push({ id: i });
    }
    // Row 261 introduces a new key
    objects.push({ id: 260, lateKey: "surprise" });
    const json = JSON.stringify(objects);
    const parsed = parseJsonFull(json);
    expect(parsed.schema.names).toContain("lateKey");
    // The lateKey column should have null for first 260 rows, "surprise" for last
    const lateIdx = parsed.schema.names.indexOf("lateKey");
    expect(parsed.columns[lateIdx][260]).toBe("surprise");
    expect(parsed.columns[lateIdx][0]).toBeNull();
  });

  it("detects float type from late rows", () => {
    const objects: Record<string, unknown>[] = [];
    for (let i = 0; i < 260; i++) {
      objects.push({ val: i }); // all integers
    }
    objects.push({ val: 3.14 }); // float at row 261
    const json = JSON.stringify(objects);
    const parsed = parseJsonFull(json);
    expect(parsed.schema.types[0]).toBe("float64");
    expect(parsed.columns[0][260]).toBe(3.14);
  });

  it("NDJSON also uses full inference", () => {
    const lines: string[] = [];
    for (let i = 0; i < 260; i++) {
      lines.push(JSON.stringify({ n: i }));
    }
    lines.push(JSON.stringify({ n: 1.5 }));
    const ndjson = lines.join("\n");
    const parsed = parseJsonFull(ndjson);
    expect(parsed.schema.types[0]).toBe("float64");
  });
});
