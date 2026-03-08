/** Runtime SQL expression evaluator — evaluates AST nodes against Row data */

import type { Row } from "../types.js";
import type { SqlExpr } from "./ast.js";

export function evaluateExpr(expr: SqlExpr, row: Row): unknown {
  switch (expr.kind) {
    case "value":
      if (expr.value.type === "null") return null;
      return expr.value.value;

    case "column":
      // Look up column name in row (ignore table qualifier — rows are flat)
      return row[expr.name] ?? null;

    case "star":
      return null;

    case "near":
      return true; // NEAR is handled by vectorSearch, always pass through

    case "binary":
      return evaluateBinary(expr.op, expr.left, expr.right, row);

    case "unary":
      return evaluateUnary(expr.op, expr.operand, row);

    case "in_list": {
      const val = evaluateExpr(expr.expr, row);
      if (val === null) return null;
      const values = expr.values.map(v => evaluateExpr(v, row));
      const found = values.some(v => looseEqual(val, v));
      return expr.negated ? !found : found;
    }

    case "between": {
      const val = evaluateExpr(expr.expr, row);
      const low = evaluateExpr(expr.low, row);
      const high = evaluateExpr(expr.high, row);
      if (val === null || low === null || high === null) return null;
      return compare(val, low) >= 0 && compare(val, high) <= 0;
    }

    case "case_expr": {
      if (expr.operand) {
        // Simple CASE: CASE expr WHEN val THEN result
        const operand = evaluateExpr(expr.operand, row);
        for (const { condition, result } of expr.whenClauses) {
          if (looseEqual(operand, evaluateExpr(condition, row))) {
            return evaluateExpr(result, row);
          }
        }
      } else {
        // Searched CASE: CASE WHEN condition THEN result
        for (const { condition, result } of expr.whenClauses) {
          if (isTruthy(evaluateExpr(condition, row))) {
            return evaluateExpr(result, row);
          }
        }
      }
      return expr.elseResult ? evaluateExpr(expr.elseResult, row) : null;
    }

    case "cast": {
      const val = evaluateExpr(expr.expr, row);
      return castValue(val, expr.targetType);
    }

    case "call":
      // Aggregate/window calls can't be evaluated row-by-row.
      // In HAVING context, aggregate calls are rewritten to column refs before evaluation.
      return null;
  }
}

export function isTruthy(val: unknown): boolean {
  return val !== null && val !== undefined && val !== false && val !== 0;
}

function evaluateBinary(op: string, leftExpr: SqlExpr, rightExpr: SqlExpr, row: Row): unknown {
  // SQL three-valued logic for AND/OR
  if (op === "and") {
    const left = evaluateExpr(leftExpr, row);
    const right = evaluateExpr(rightExpr, row);
    if (left === false || left === 0) return false;
    if (right === false || right === 0) return false;
    if (left === null || right === null) return null;
    return isTruthy(left) && isTruthy(right);
  }
  if (op === "or") {
    const left = evaluateExpr(leftExpr, row);
    const right = evaluateExpr(rightExpr, row);
    if (isTruthy(left) || isTruthy(right)) return true;
    if (left === null || right === null) return null;
    return false;
  }

  const left = evaluateExpr(leftExpr, row);
  const right = evaluateExpr(rightExpr, row);

  // SQL null propagation: any op with null returns null (except AND/OR handled above)
  if (left === null || right === null) {
    return null;
  }

  switch (op) {
    case "eq": return looseEqual(left, right);
    case "ne": return !looseEqual(left, right);
    case "lt": return compare(left, right) < 0;
    case "le": return compare(left, right) <= 0;
    case "gt": return compare(left, right) > 0;
    case "ge": return compare(left, right) >= 0;
    case "add": return toNumber(left) + toNumber(right);
    case "subtract": return toNumber(left) - toNumber(right);
    case "multiply": return toNumber(left) * toNumber(right);
    case "divide": {
      const divisor = toNumber(right);
      return divisor === 0 ? null : toNumber(left) / divisor;
    }
    case "concat": return String(left) + String(right);
    case "like": return matchLike(String(left), String(right));
    default: return null;
  }
}

function evaluateUnary(op: string, operandExpr: SqlExpr, row: Row): unknown {
  if (op === "is_null") {
    const val = evaluateExpr(operandExpr, row);
    return val === null || val === undefined;
  }
  if (op === "is_not_null") {
    const val = evaluateExpr(operandExpr, row);
    return val !== null && val !== undefined;
  }

  const val = evaluateExpr(operandExpr, row);

  if (op === "not") {
    if (val === null) return null;
    return !isTruthy(val);
  }
  if (op === "minus") {
    if (val === null) return null;
    return -toNumber(val);
  }
  return null;
}

function looseEqual(a: unknown, b: unknown): boolean {
  if (typeof a === "number" && typeof b === "number") return a === b;
  if (typeof a === "bigint" && typeof b === "bigint") return a === b;
  if (typeof a === "string" && typeof b === "string") return a === b;
  if (typeof a === "boolean" && typeof b === "boolean") return a === b;
  // Cross-type numeric comparison
  if (typeof a === "number" && typeof b === "bigint") return a === Number(b);
  if (typeof a === "bigint" && typeof b === "number") return Number(a) === b;
  return a === b;
}

function compare(a: unknown, b: unknown): number {
  if (typeof a === "number" && typeof b === "number") return a - b;
  if (typeof a === "bigint" && typeof b === "bigint") return a < b ? -1 : a > b ? 1 : 0;
  if (typeof a === "string" && typeof b === "string") return a.localeCompare(b);
  // Cross-type: coerce to number
  return toNumber(a) - toNumber(b);
}

function toNumber(val: unknown): number {
  if (typeof val === "number") return val;
  if (typeof val === "bigint") return Number(val);
  if (typeof val === "string") return parseFloat(val) || 0;
  if (typeof val === "boolean") return val ? 1 : 0;
  return 0;
}

/** Convert SQL LIKE pattern to case-insensitive match. % = any chars, _ = one char. */
function matchLike(value: string, pattern: string): boolean {
  let re = "^";
  for (let i = 0; i < pattern.length; i++) {
    const ch = pattern[i];
    if (ch === "%") re += ".*";
    else if (ch === "_") re += ".";
    else if (/[.*+?^${}()|[\]\\]/.test(ch)) re += "\\" + ch;
    else re += ch;
  }
  re += "$";
  return new RegExp(re, "i").test(value);
}

function castValue(val: unknown, targetType: string): unknown {
  if (val === null) return null;
  const t = targetType.toLowerCase();
  if (t === "int" || t === "integer" || t === "bigint") return Math.trunc(toNumber(val));
  if (t === "float" || t === "double" || t === "real" || t === "decimal" || t === "numeric") return toNumber(val);
  if (t === "text" || t === "varchar" || t === "string" || t === "char") return String(val);
  if (t === "bool" || t === "boolean") return isTruthy(val);
  return val;
}

/**
 * Rewrite aggregate calls in a HAVING expression to column references.
 * e.g. COUNT(*) > 5 → column("count_*") > 5
 * This lets the evaluator look up aggregate results by their output column name.
 */
export function rewriteAggregatesAsColumns(expr: SqlExpr): SqlExpr {
  switch (expr.kind) {
    case "call": {
      const fnUpper = expr.name.toUpperCase();
      if (isAggregate(fnUpper)) {
        const col = expr.args[0]?.kind === "star" ? "*" : (expr.args[0]?.kind === "column" ? expr.args[0].name : "*");
        const fn = (expr.distinct && fnUpper === "COUNT") ? "count_distinct" : fnUpper.toLowerCase();
        return { kind: "column", name: `${fn}_${col}` };
      }
      return expr;
    }
    case "binary":
      return {
        kind: "binary", op: expr.op,
        left: rewriteAggregatesAsColumns(expr.left),
        right: rewriteAggregatesAsColumns(expr.right),
      };
    case "unary":
      return { kind: "unary", op: expr.op, operand: rewriteAggregatesAsColumns(expr.operand) };
    default:
      return expr;
  }
}

function isAggregate(name: string): boolean {
  return name === "COUNT" || name === "SUM" || name === "AVG" || name === "MIN" || name === "MAX" ||
    name === "COUNT_DISTINCT" || name === "STDDEV" || name === "VARIANCE" || name === "MEDIAN";
}
