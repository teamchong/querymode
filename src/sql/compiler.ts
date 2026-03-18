/** SQL AST → QueryDescriptor compiler */

import type { QueryDescriptor } from "../client.js";
import type { AggregateOp, FilterOp, WindowSpec } from "../types.js";
import { aggArgKey, type SelectStmt, type SqlExpr, type TableRef, type SqlOrderBy, type CteDef } from "./ast.js";
import { rewriteAggregatesAsColumns } from "./evaluator.js";

const AGGREGATE_FNS = new Set(["COUNT", "SUM", "AVG", "MIN", "MAX", "COUNT_DISTINCT", "STDDEV", "VARIANCE", "MEDIAN", "PERCENTILE"]);

/** A compiled CTE — name + its full compile result */
export interface CompiledCte {
  name: string;
  result: SqlCompileResult;
}

/** Full compile result with extras that need runtime evaluation */
export interface SqlCompileResult {
  descriptor: QueryDescriptor;
  /** Full WHERE expression — set when filters can't fully represent the WHERE clause */
  whereExpr?: SqlExpr;
  /** HAVING expression (aggregate calls rewritten to column refs for row-level evaluation) */
  havingExpr?: SqlExpr;
  /** All ORDER BY columns — set when there are multiple (descriptor only holds the first) */
  allOrderBy?: SqlOrderBy[];
  /** Non-column SELECT expressions (CASE, CAST, arithmetic) that need per-row computation */
  computedExprs?: { alias: string; expr: SqlExpr }[];
  /** CTEs that must be materialized before the main query runs */
  ctes?: CompiledCte[];
}

/** Compile to just a QueryDescriptor (backward-compatible) */
export function compile(stmt: SelectStmt): QueryDescriptor {
  return compileFull(stmt).descriptor;
}

/** Full compilation returning extras for runtime evaluation */
export function compileFull(stmt: SelectStmt): SqlCompileResult {
  // Compile CTEs first — each becomes a materialized subquery
  let compiledCtes: CompiledCte[] | undefined;
  if (stmt.ctes && stmt.ctes.length > 0) {
    compiledCtes = stmt.ctes.map(cte => ({
      name: cte.name,
      result: compileFull(cte.query),
    }));
  }

  const table = extractTableName(stmt.from);
  const filters: FilterOp[] = [];
  const aggregates: AggregateOp[] = [];
  const windows: WindowSpec[] = [];
  let projections: string[] = [];
  const computedExprs: { alias: string; expr: SqlExpr }[] = [];

  // Process SELECT columns
  let hasAggregates = false;
  let selectItemIndex = 0;
  for (const col of stmt.columns) {
    if (col.expr.kind === "star") {
      continue;
    }
    if (col.expr.kind === "call" && col.expr.window) {
      windows.push(compileWindowFunction(col.expr, col.alias));
      continue;
    }
    if (col.expr.kind === "call" && isAggregateCall(col.expr.name)) {
      hasAggregates = true;
      aggregates.push(compileAggregate(col.expr, col.alias));
      continue;
    }
    // Simple column reference → projection
    const colName = extractColumnName(col.expr);
    if (colName) {
      projections.push(colName);
    } else {
      // Non-column expression (CASE, CAST, arithmetic, etc.) → computed expression
      const alias = col.alias ?? `_expr${selectItemIndex}`;
      computedExprs.push({ alias, expr: col.expr });
    }
    selectItemIndex++;
  }

  // If there are aggregates, projections should only contain group-by columns
  if (hasAggregates && stmt.groupBy) {
    projections = [...stmt.groupBy.columns];
  }

  // Process WHERE clause — try to flatten to FilterOp[] (and filterGroups for OR)
  let whereExpr: SqlExpr | undefined;
  let filterGroups: FilterOp[][] | undefined;
  if (stmt.where) {
    const savedLen = filters.length;
    const fullyFlattened = tryFlattenFilters(stmt.where, filters);
    if (!fullyFlattened) {
      // Roll back any partially-pushed filters to avoid double-filtering
      filters.length = savedLen;
      // Try OR decomposition: if top-level is OR, flatten each branch independently
      const orGroups = tryFlattenOrGroups(stmt.where);
      if (orGroups) {
        filterGroups = orGroups;
      } else {
        // Some predicates couldn't be pushed down — set whereExpr for runtime evaluation
        whereExpr = stmt.where;
      }
    }
  }

  // Process vector search (NEAR in WHERE)
  let vectorSearch: QueryDescriptor["vectorSearch"];
  if (stmt.where) {
    vectorSearch = extractVectorSearch(stmt.where);
  }

  // ORDER BY
  let sortColumn: string | undefined;
  let sortDirection: "asc" | "desc" | undefined;
  let allOrderBy: SqlOrderBy[] | undefined;

  if (stmt.orderBy && stmt.orderBy.length > 0) {
    if (stmt.orderBy.length === 1) {
      // Single column — use descriptor's native sort
      sortColumn = stmt.orderBy[0].column;
      sortDirection = stmt.orderBy[0].direction;
    } else {
      // Multi-column — wrapper handles sort (don't set sortColumn to avoid incorrect limit)
      allOrderBy = stmt.orderBy;
    }
  }

  // JOIN
  let join: QueryDescriptor["join"];
  if (stmt.from.kind === "join") {
    join = compileJoin(stmt.from);
  }

  // Set operation
  let setOperation: QueryDescriptor["setOperation"];
  if (stmt.setOperation) {
    const rightDesc = compile(stmt.setOperation.right);
    const modeMap = {
      union_all: "union_all" as const,
      union_distinct: "union" as const,
      intersect: "intersect" as const,
      except: "except" as const,
    };
    setOperation = { mode: modeMap[stmt.setOperation.opType], right: rightDesc };
  }

  // DISTINCT
  let distinct: string[] | undefined;
  if (stmt.distinct) {
    distinct = projections.length > 0 ? projections : [];
  }

  // HAVING — rewrite aggregate calls to column refs for post-aggregation evaluation
  let havingExpr: SqlExpr | undefined;
  if (stmt.groupBy?.having) {
    havingExpr = rewriteAggregatesAsColumns(stmt.groupBy.having);
  }

  const desc: QueryDescriptor = {
    table,
    filters,
    filterGroups,
    projections: projections.length > 0 ? projections : [],
    sortColumn,
    sortDirection,
    limit: stmt.limit,
    offset: stmt.offset,
    aggregates: aggregates.length > 0 ? aggregates : undefined,
    groupBy: stmt.groupBy ? stmt.groupBy.columns : undefined,
    vectorSearch,
    join,
    setOperation,
    distinct,
    windows: windows.length > 0 ? windows : undefined,
  };

  // Inline CTE: if FROM references a CTE name, merge its filters and resolve to the real table.
  // Only inline simple CTEs (filter-only). CTEs with sort/limit/aggregates need materialization.
  if (compiledCtes) {
    const cteMap = new Map(compiledCtes.map(c => [c.name, c.result]));
    const cte = cteMap.get(desc.table);
    if (cte) {
      const base = cte.descriptor;
      // Guard: CTEs with sort/limit/aggregates/groupBy/having/distinct/windows
      // can't be inlined — they change the result shape and need materialization first.
      const canInline = !base.sortColumn && base.limit === undefined &&
        !base.aggregates && !base.groupBy && !cte.havingExpr && !cte.allOrderBy &&
        !(base.distinct && base.distinct.length > 0) &&
        !(base.windows && base.windows.length > 0) &&
        !(base.filterGroups && base.filterGroups.length > 0 && desc.filterGroups && desc.filterGroups.length > 0);
      if (canInline) {
        desc.table = base.table;
        desc.filters = [...base.filters, ...desc.filters];
        desc.filterGroups = [...(base.filterGroups ?? []), ...(desc.filterGroups ?? [])];
        // Merge CTE's residual whereExpr into outer whereExpr via AND
        if (cte.whereExpr) {
          whereExpr = whereExpr
            ? { kind: "binary", op: "and", left: cte.whereExpr, right: whereExpr } as SqlExpr
            : cte.whereExpr;
        }
        // Propagate CTE's computedExprs
        if (cte.computedExprs) {
          computedExprs.unshift(...cte.computedExprs);
        }
      }
      // If !canInline, table name stays as CTE name — will fail at execution
      // (CTE materialization not yet supported)
    }
  }

  return {
    descriptor: desc,
    whereExpr,
    havingExpr,
    allOrderBy,
    computedExprs: computedExprs.length > 0 ? computedExprs : undefined,
    ctes: compiledCtes,
  };
}

function extractTableName(from: TableRef): string {
  if (from.kind === "simple") return from.name;
  if (from.kind === "join") return extractTableName(from.left);
  return "unknown";
}

function extractColumnName(expr: SqlExpr): string | undefined {
  if (expr.kind === "column") return expr.name;
  return undefined;
}


function isAggregateCall(name: string): boolean {
  return AGGREGATE_FNS.has(name.toUpperCase());
}

function compileAggregate(expr: SqlExpr & { kind: "call" }, alias?: string): AggregateOp {
  let fn = expr.name.toLowerCase() as AggregateOp["fn"];
  let column = "*";

  if (fn === "count" && expr.distinct && expr.args.length > 0) {
    fn = "count_distinct";
  }

  if (expr.args.length > 0 && expr.args[0].kind !== "star") {
    column = aggArgKey(expr.args[0]);
  }

  const result: AggregateOp = { fn, column };
  if (alias) result.alias = alias;

  if (fn === "percentile" && expr.args.length >= 2) {
    const pArg = expr.args[1];
    if (pArg.kind === "value" && (pArg.value.type === "float" || pArg.value.type === "integer")) {
      result.percentileTarget = pArg.value.value;
    }
  }

  return result;
}

function compileWindowFunction(expr: SqlExpr & { kind: "call" }, alias?: string): WindowSpec {
  const fn = expr.name.toLowerCase() as WindowSpec["fn"];
  const win = expr.window!;

  const spec: WindowSpec = {
    fn,
    partitionBy: win.partitionBy ?? [],
    orderBy: (win.orderBy ?? []).map(o => ({ column: o.column, direction: o.direction })),
    alias: alias ?? `${fn}_result`,
  };

  if (expr.args.length > 0) {
    const colArg = expr.args[0];
    if (colArg.kind === "column") {
      spec.column = colArg.name;
    } else if (colArg.kind === "star") {
      spec.column = "*";
    }
  }

  if ((fn === "lag" || fn === "lead") && expr.args.length > 1) {
    const args: WindowSpec["args"] = {};
    if (expr.args[1]?.kind === "value" && expr.args[1].value.type === "integer") {
      args.offset = expr.args[1].value.value;
    }
    if (expr.args[2]?.kind === "value") {
      args.default_ = extractLiteralValue(expr.args[2]);
    }
    spec.args = args;
  }

  if (win.frame) {
    const frameBoundToValue = (b: NonNullable<typeof win.frame>["start"]): number | "unbounded" | "current" => {
      if (b.type === "unbounded_preceding" || b.type === "unbounded_following") return "unbounded";
      if (b.type === "current_row") return "current";
      if (b.type === "preceding") return -b.offset;
      return b.offset;
    };

    spec.frame = {
      type: win.frame.type,
      start: frameBoundToValue(win.frame.start),
      end: win.frame.end ? frameBoundToValue(win.frame.end) : "current",
    };
  }

  return spec;
}

/**
 * Try to flatten a WHERE expression into AND-connected FilterOps.
 * Returns true if fully flattened, false if some predicates were skipped.
 */
function tryFlattenFilters(expr: SqlExpr, filters: FilterOp[]): boolean {
  if (expr.kind === "binary" && expr.op === "and") {
    const leftOk = tryFlattenFilters(expr.left, filters);
    const rightOk = tryFlattenFilters(expr.right, filters);
    return leftOk && rightOk;
  }

  // Simple comparison: column op value
  if (expr.kind === "binary" && isComparisonOp(expr.op)) {
    const filter = compileSimpleFilter(expr);
    if (filter) {
      filters.push(filter);
      return true;
    }
    return false;
  }

  // IS NULL / IS NOT NULL
  if (expr.kind === "unary" && (expr.op === "is_null" || expr.op === "is_not_null")) {
    const colName = extractColumnName(expr.operand);
    if (colName) {
      filters.push({ column: colName, op: expr.op, value: 0 });
      return true;
    }
    return false;
  }

  // IN list: column IN (v1, v2, ...) or column NOT IN (v1, v2, ...)
  if (expr.kind === "in_list") {
    const colName = extractColumnName(expr.expr);
    if (colName) {
      const values = expr.values.map(extractLiteralValue).filter((v): v is NonNullable<typeof v> => v !== undefined);
      if (values.length === expr.values.length) {
        filters.push({ column: colName, op: expr.negated ? "not_in" : "in", value: values as (number | string)[] });
        return true;
      }
    }
    return false;
  }

  // BETWEEN: column BETWEEN low AND high (NOT BETWEEN is handled via unary NOT wrapping below)
  if (expr.kind === "between") {
    const colName = extractColumnName(expr.expr);
    const low = extractLiteralValue(expr.low);
    const high = extractLiteralValue(expr.high);
    if (colName && low !== undefined && high !== undefined) {
      filters.push({ column: colName, op: "between", value: [low as number | string, high as number | string] });
      return true;
    }
    return false;
  }

  // LIKE: column LIKE 'pattern'
  if (expr.kind === "binary" && expr.op === "like") {
    const colName = extractColumnName(expr.left);
    const pattern = extractLiteralValue(expr.right);
    if (colName && typeof pattern === "string") {
      filters.push({ column: colName, op: "like", value: pattern });
      return true;
    }
    return false;
  }

  // NOT (BETWEEN | LIKE): { kind: "unary", op: "not", operand: ... }
  if (expr.kind === "unary" && expr.op === "not") {
    const inner = expr.operand;
    // NOT BETWEEN
    if (inner.kind === "between") {
      const colName = extractColumnName(inner.expr);
      const low = extractLiteralValue(inner.low);
      const high = extractLiteralValue(inner.high);
      if (colName && low !== undefined && high !== undefined) {
        filters.push({ column: colName, op: "not_between", value: [low as number | string, high as number | string] });
        return true;
      }
      return false;
    }
    // NOT LIKE
    if (inner.kind === "binary" && inner.op === "like") {
      const colName = extractColumnName(inner.left);
      const pattern = extractLiteralValue(inner.right);
      if (colName && typeof pattern === "string") {
        filters.push({ column: colName, op: "not_like", value: pattern });
        return true;
      }
      return false;
    }
    // NOT IN is already handled by in_list.negated
  }

  // NEAR expressions are handled separately by extractVectorSearch
  if (expr.kind === "near") {
    return true;
  }

  // Everything else (OR, subqueries, etc.) can't be flattened
  return false;
}

/**
 * Try to decompose a WHERE expression into OR-connected groups.
 * Returns FilterOp[][] where each inner array is AND-connected, or null if not possible.
 * Handles: (a AND b) OR (c AND d) OR e
 */
function tryFlattenOrGroups(expr: SqlExpr): FilterOp[][] | null {
  // Collect OR branches
  const branches: SqlExpr[] = [];
  collectOrBranches(expr, branches);
  if (branches.length < 2) return null; // not an OR expression

  const groups: FilterOp[][] = [];
  for (const branch of branches) {
    const filters: FilterOp[] = [];
    if (tryFlattenFilters(branch, filters) && filters.length > 0) {
      groups.push(filters);
    } else {
      return null; // can't flatten this branch — bail
    }
  }
  return groups;
}

/** Recursively collect OR branches from a tree of OR expressions. */
function collectOrBranches(expr: SqlExpr, out: SqlExpr[]): void {
  if (expr.kind === "binary" && expr.op === "or") {
    collectOrBranches(expr.left, out);
    collectOrBranches(expr.right, out);
  } else {
    out.push(expr);
  }
}

function isComparisonOp(op: string): boolean {
  return op === "eq" || op === "ne" || op === "lt" || op === "le" || op === "gt" || op === "ge";
}

function compileSimpleFilter(expr: SqlExpr & { kind: "binary" }): FilterOp | undefined {
  const colName = extractColumnName(expr.left);
  const value = extractLiteralValue(expr.right);

  const opMap: Record<string, FilterOp["op"]> = {
    eq: "eq", ne: "neq", lt: "lt", le: "lte", gt: "gt", ge: "gte",
  };

  if (!colName || value === undefined) {
    const rCol = extractColumnName(expr.right);
    const lVal = extractLiteralValue(expr.left);
    if (rCol && lVal !== undefined) {
      const flippedOp = flipOp(expr.op);
      const filterOp = flippedOp ? opMap[flippedOp] : undefined;
      if (filterOp) {
        return { column: rCol, op: filterOp, value: lVal as FilterOp["value"] };
      }
    }
    return undefined;
  }

  const filterOp = opMap[expr.op];
  if (!filterOp) return undefined;

  return { column: colName, op: filterOp, value: value as FilterOp["value"] };
}

function flipOp(op: string): string | undefined {
  const map: Record<string, string> = { lt: "gt", le: "ge", gt: "lt", ge: "le", eq: "eq", ne: "ne" };
  return map[op];
}

function extractLiteralValue(expr: SqlExpr): number | string | boolean | undefined {
  if (expr.kind === "value") {
    if (expr.value.type === "integer" || expr.value.type === "float") return expr.value.value;
    if (expr.value.type === "string") return expr.value.value;
    if (expr.value.type === "boolean") return expr.value.value;
  }
  if (expr.kind === "unary" && expr.op === "minus" && expr.operand.kind === "value") {
    const v = expr.operand.value;
    if (v.type === "integer" || v.type === "float") return -v.value;
  }
  return undefined;
}

function extractVectorSearch(expr: SqlExpr): QueryDescriptor["vectorSearch"] {
  if (expr.kind === "near") {
    const colName = extractColumnName(expr.column);
    if (colName) {
      return {
        column: colName,
        queryVector: new Float32Array(expr.vector),
        topK: expr.topK ?? 10,
      };
    }
  }
  if (expr.kind === "binary" && expr.op === "and") {
    return extractVectorSearch(expr.left) ?? extractVectorSearch(expr.right);
  }
  return undefined;
}

function compileJoin(ref: TableRef & { kind: "join" }): QueryDescriptor["join"] {
  const joinClause = ref.join;
  const rightTable = joinClause.table.kind === "simple" ? joinClause.table.name : "unknown";

  let leftKey = "";
  let rightKey = "";
  if (joinClause.onCondition && joinClause.onCondition.kind === "binary" && joinClause.onCondition.op === "eq") {
    const lCol = joinClause.onCondition.left;
    const rCol = joinClause.onCondition.right;
    if (lCol.kind === "column") leftKey = lCol.name;
    if (rCol.kind === "column") rightKey = rCol.name;
  }

  return {
    right: {
      table: rightTable,
      filters: [],
      projections: [],
    },
    leftKey,
    rightKey,
    type: joinClause.joinType,
  };
}
