import { describe, it, expect } from "vitest";
import { evaluateExpr, isTruthy, rewriteAggregatesAsColumns } from "./evaluator.js";
import type { SqlExpr } from "./ast.js";

const row = { name: "Alice", age: 30, dept: "eng", salary: 120000, notes: null };

function col(name: string): SqlExpr {
  return { kind: "column", name };
}
function val(v: number | string | boolean): SqlExpr {
  if (typeof v === "number") {
    return Number.isInteger(v)
      ? { kind: "value", value: { type: "integer", value: v } }
      : { kind: "value", value: { type: "float", value: v } };
  }
  if (typeof v === "string") return { kind: "value", value: { type: "string", value: v } };
  return { kind: "value", value: { type: "boolean", value: v } };
}
function bin(op: string, left: SqlExpr, right: SqlExpr): SqlExpr {
  return { kind: "binary", op, left, right };
}

describe("SQL Evaluator", () => {
  it("evaluates column lookup", () => {
    expect(evaluateExpr(col("age"), row)).toBe(30);
    expect(evaluateExpr(col("name"), row)).toBe("Alice");
  });

  it("evaluates literal values", () => {
    expect(evaluateExpr(val(42), row)).toBe(42);
    expect(evaluateExpr(val("hello"), row)).toBe("hello");
    expect(evaluateExpr(val(true), row)).toBe(true);
  });

  it("evaluates comparisons", () => {
    expect(evaluateExpr(bin("gt", col("age"), val(25)), row)).toBe(true);
    expect(evaluateExpr(bin("lt", col("age"), val(25)), row)).toBe(false);
    expect(evaluateExpr(bin("eq", col("dept"), val("eng")), row)).toBe(true);
    expect(evaluateExpr(bin("ne", col("dept"), val("sales")), row)).toBe(true);
  });

  it("evaluates AND", () => {
    const expr = bin("and", bin("gt", col("age"), val(25)), bin("eq", col("dept"), val("eng")));
    expect(evaluateExpr(expr, row)).toBe(true);
    const expr2 = bin("and", bin("gt", col("age"), val(35)), bin("eq", col("dept"), val("eng")));
    expect(evaluateExpr(expr2, row)).toBe(false);
  });

  it("evaluates OR", () => {
    const expr = bin("or", bin("gt", col("age"), val(35)), bin("eq", col("dept"), val("eng")));
    expect(evaluateExpr(expr, row)).toBe(true);
    const expr2 = bin("or", bin("gt", col("age"), val(35)), bin("eq", col("dept"), val("sales")));
    expect(evaluateExpr(expr2, row)).toBe(false);
  });

  it("evaluates arithmetic", () => {
    expect(evaluateExpr(bin("add", col("age"), val(10)), row)).toBe(40);
    expect(evaluateExpr(bin("subtract", col("age"), val(5)), row)).toBe(25);
    expect(evaluateExpr(bin("multiply", val(3), val(4)), row)).toBe(12);
    expect(evaluateExpr(bin("divide", col("salary"), val(1000)), row)).toBe(120);
  });

  it("evaluates division by zero as null", () => {
    expect(evaluateExpr(bin("divide", val(10), val(0)), row)).toBe(null);
  });

  it("evaluates LIKE", () => {
    expect(evaluateExpr(bin("like", col("name"), val("%lic%")), row)).toBe(true);
    expect(evaluateExpr(bin("like", col("name"), val("Ali__")), row)).toBe(true);
    expect(evaluateExpr(bin("like", col("name"), val("Bob%")), row)).toBe(false);
  });

  it("evaluates concat", () => {
    expect(evaluateExpr(bin("concat", col("name"), val("!")), row)).toBe("Alice!");
  });

  it("evaluates IS NULL / IS NOT NULL", () => {
    const isNull: SqlExpr = { kind: "unary", op: "is_null", operand: col("notes") };
    const isNotNull: SqlExpr = { kind: "unary", op: "is_not_null", operand: col("name") };
    expect(evaluateExpr(isNull, row)).toBe(true);
    expect(evaluateExpr(isNotNull, row)).toBe(true);
  });

  it("evaluates NOT", () => {
    expect(evaluateExpr({ kind: "unary", op: "not", operand: val(true) }, row)).toBe(false);
    expect(evaluateExpr({ kind: "unary", op: "not", operand: val(false) }, row)).toBe(true);
  });

  it("evaluates IN list", () => {
    const expr: SqlExpr = { kind: "in_list", expr: col("dept"), values: [val("eng"), val("sales")], negated: false };
    expect(evaluateExpr(expr, row)).toBe(true);
  });

  it("evaluates NOT IN list", () => {
    const expr: SqlExpr = { kind: "in_list", expr: col("dept"), values: [val("sales"), val("hr")], negated: true };
    expect(evaluateExpr(expr, row)).toBe(true);
    const expr2: SqlExpr = { kind: "in_list", expr: col("dept"), values: [val("eng"), val("hr")], negated: true };
    expect(evaluateExpr(expr2, row)).toBe(false);
  });

  it("evaluates BETWEEN", () => {
    const expr: SqlExpr = { kind: "between", expr: col("age"), low: val(25), high: val(35) };
    expect(evaluateExpr(expr, row)).toBe(true);
    const expr2: SqlExpr = { kind: "between", expr: col("age"), low: val(31), high: val(35) };
    expect(evaluateExpr(expr2, row)).toBe(false);
  });

  it("evaluates CASE WHEN", () => {
    const expr: SqlExpr = {
      kind: "case_expr",
      whenClauses: [
        { condition: bin("gt", col("age"), val(25)), result: val("senior") },
      ],
      elseResult: val("junior"),
    };
    expect(evaluateExpr(expr, row)).toBe("senior");
    expect(evaluateExpr(expr, { ...row, age: 20 })).toBe("junior");
  });

  it("evaluates CAST", () => {
    const expr: SqlExpr = { kind: "cast", expr: col("age"), targetType: "float" };
    expect(evaluateExpr(expr, row)).toBe(30);
    const expr2: SqlExpr = { kind: "cast", expr: col("age"), targetType: "text" };
    expect(evaluateExpr(expr2, row)).toBe("30");
  });

  it("propagates null in binary ops", () => {
    expect(evaluateExpr(bin("add", col("notes"), val(1)), row)).toBe(null);
    expect(evaluateExpr(bin("eq", col("notes"), val(1)), row)).toBe(null);
  });
});

describe("isTruthy", () => {
  it("returns false for null, undefined, false, 0", () => {
    expect(isTruthy(null)).toBe(false);
    expect(isTruthy(undefined)).toBe(false);
    expect(isTruthy(false)).toBe(false);
    expect(isTruthy(0)).toBe(false);
  });

  it("returns true for truthy values", () => {
    expect(isTruthy(1)).toBe(true);
    expect(isTruthy("hello")).toBe(true);
    expect(isTruthy(true)).toBe(true);
  });
});

describe("rewriteAggregatesAsColumns", () => {
  it("rewrites COUNT(*) to column ref", () => {
    const expr: SqlExpr = { kind: "call", name: "COUNT", args: [{ kind: "star" }], distinct: false };
    const result = rewriteAggregatesAsColumns(expr);
    expect(result).toEqual({ kind: "column", name: "count_*" });
  });

  it("rewrites SUM(col) to column ref", () => {
    const expr: SqlExpr = { kind: "call", name: "SUM", args: [col("amount")], distinct: false };
    const result = rewriteAggregatesAsColumns(expr);
    expect(result).toEqual({ kind: "column", name: "sum_amount" });
  });

  it("rewrites nested binary with aggregates", () => {
    const expr = bin("gt",
      { kind: "call", name: "COUNT", args: [{ kind: "star" }], distinct: false },
      val(5),
    );
    const result = rewriteAggregatesAsColumns(expr);
    expect(result.kind).toBe("binary");
    if (result.kind === "binary") {
      expect(result.left).toEqual({ kind: "column", name: "count_*" });
    }
  });
});
