import { describe, it, expect } from "vitest";
import { createFromJSON } from "./convenience.js";
import { DataFrame, MaterializedExecutor } from "./client.js";
import { ComputedColumnOperator, FilterOperator, drainPipeline } from "./operators.js";
import type { Operator, RowBatch } from "./operators.js";
import type { Row } from "./types.js";

/** Simple operator that yields a fixed array of rows in one batch. */
class ArrayOperator implements Operator {
  private rows: Row[];
  private done = false;
  constructor(rows: Row[]) {
    this.rows = rows;
  }
  async next(): Promise<RowBatch | null> {
    if (this.done) return null;
    this.done = true;
    return this.rows;
  }
  async close() {}
}

const testData = [
  { id: 1, name: "Alice", age: 30 },
  { id: 2, name: "Bob", age: 25 },
  { id: 3, name: "Charlie", age: 35 },
];

const dummyExecutor = new MaterializedExecutor({
  rows: [],
  rowCount: 0,
  columns: [],
  bytesRead: 0,
  pagesSkipped: 0,
  durationMs: 0,
});

describe(".pipe()", () => {
  it("injects a ComputedColumnOperator that adds a column", async () => {
    const op = new ArrayOperator([...testData.map((r) => ({ ...r }))]);
    const computed = new ComputedColumnOperator(op, [
      { alias: "senior", fn: (row) => (row.age as number) >= 30 },
    ]);
    const rows = await drainPipeline(computed);
    expect(rows.length).toBe(3);
    expect(rows[0]).toHaveProperty("senior", true);
    expect(rows[1]).toHaveProperty("senior", false);
    expect(rows[2]).toHaveProperty("senior", true);
  });

  it("chains multiple pipe stages in order", async () => {
    const op = new ArrayOperator([...testData.map((r) => ({ ...r }))]);
    const stage1 = new ComputedColumnOperator(op, [
      { alias: "doubled", fn: (row) => (row.age as number) * 2 },
    ]);
    const stage2 = new ComputedColumnOperator(stage1, [
      { alias: "tripled", fn: (row) => (row.doubled as number) + (row.age as number) },
    ]);
    const rows = await drainPipeline(stage2);
    expect(rows.length).toBe(3);
    // Alice: doubled=60, tripled=60+30=90
    expect(rows[0].doubled).toBe(60);
    expect(rows[0].tripled).toBe(90);
  });

  it("filter after pipe applies to computed column", async () => {
    const op = new ArrayOperator([...testData.map((r) => ({ ...r }))]);
    const computed = new ComputedColumnOperator(op, [
      { alias: "score", fn: (row) => (row.age as number) * 10 },
    ]);
    const filtered = new FilterOperator(computed, [
      { column: "score", op: "gte", value: 300 },
    ]);
    const rows = await drainPipeline(filtered);
    expect(rows.length).toBe(2);
    expect(rows.every((r) => (r.score as number) >= 300)).toBe(true);
  });

  it("returns a new DataFrame -- original is unchanged", () => {
    const original = createFromJSON(testData);
    const piped = original.pipe(
      (upstream) =>
        new ComputedColumnOperator(upstream, [
          { alias: "extra", fn: () => 42 },
        ]),
    );
    // pipe() returns a new instance, not the same reference
    expect(piped).not.toBe(original);

    // The original descriptor should have no pipeStages
    const origDesc = original.toDescriptor();
    const pipedDesc = piped.toDescriptor();
    expect(origDesc.pipeStages).toBeUndefined();
    expect(pipedDesc.pipeStages).toHaveLength(1);
  });
});

describe("DataFrame.fromOperator()", () => {
  it("wraps a raw Operator with filtering applied before re-entry", async () => {
    const rows: Row[] = [
      { x: 1, label: "a" },
      { x: 2, label: "b" },
      { x: 3, label: "c" },
      { x: 4, label: "d" },
    ];
    // Apply filter at the operator level, then re-enter via fromOperator
    const source = new ArrayOperator(rows);
    const filtered = new FilterOperator(source, [
      { column: "x", op: "gt", value: 2 },
    ]);

    const result = await DataFrame.fromOperator(filtered, dummyExecutor).collect();

    expect(result.rowCount).toBe(2);
    expect(result.rows[0]).toEqual({ x: 3, label: "c" });
    expect(result.rows[1]).toEqual({ x: 4, label: "d" });
  });

  it("drains all rows when no further chaining", async () => {
    const rows: Row[] = [
      { a: 10 },
      { a: 20 },
      { a: 30 },
    ];
    const op = new ArrayOperator(rows);

    const result = await DataFrame.fromOperator(op, dummyExecutor).collect();

    expect(result.rowCount).toBe(3);
    expect(result.columns).toEqual(["a"]);
    expect(result.rows).toEqual([{ a: 10 }, { a: 20 }, { a: 30 }]);
  });
});
