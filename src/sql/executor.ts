/**
 * SqlWrappingExecutor — applies SQL features that can't be pushed down to FilterOp[].
 *
 * Handles: OR conditions, HAVING, multi-column ORDER BY,
 * and computed SELECT expressions (CASE, CAST, arithmetic).
 *
 * LIKE, NOT IN, NOT BETWEEN are now flattened into FilterOp[] by the compiler.
 *
 * The inner executor still receives FilterOp[] for page-level pushdown optimization.
 * This executor applies the remaining logic on the result rows.
 */

import type { QueryDescriptor, QueryExecutor } from "../client.js";
import type { AggregateOp, QueryResult, Row } from "../types.js";
import type { SqlExpr, SqlOrderBy } from "./ast.js";
import { evaluateExpr, isTruthy } from "./evaluator.js";

export interface SqlExecutorOpts {
  /** Full WHERE expression for row-level evaluation (when FilterOp[] is incomplete) */
  whereExpr?: SqlExpr;
  /** HAVING expression (rewritten: aggregate calls replaced with column refs) */
  havingExpr?: SqlExpr;
  /** All ORDER BY columns when there are multiple (single-column uses descriptor.sortColumn) */
  allOrderBy?: SqlOrderBy[];
  /** Non-column SELECT expressions that need per-row computation */
  computedExprs?: { alias: string; expr: SqlExpr }[];
}

export class SqlWrappingExecutor implements QueryExecutor {
  constructor(
    private inner: QueryExecutor,
    private opts: SqlExecutorOpts,
  ) {}

  async execute(query: QueryDescriptor): Promise<QueryResult> {
    const hasMultiSort = this.opts.allOrderBy && this.opts.allOrderBy.length > 1;
    const hasWhereExpr = !!this.opts.whereExpr;
    const hasAggregates = query.aggregates && query.aggregates.length > 0;

    // Build inner query — strip things we handle ourselves
    const innerQuery = { ...query };

    // If we have a whereExpr AND aggregates, we must get raw rows first,
    // then filter, then aggregate manually. Otherwise the inner executor
    // aggregates unfiltered rows.
    let manualAggregation = false;
    let savedAggregates: AggregateOp[] | undefined;
    let savedGroupBy: string[] | undefined;
    if (hasWhereExpr && hasAggregates) {
      manualAggregation = true;
      savedAggregates = innerQuery.aggregates;
      savedGroupBy = innerQuery.groupBy;
      delete innerQuery.aggregates;
      delete innerQuery.groupBy;
    }

    // If we have computed expressions, we need all columns for evaluation
    // — strip projections from inner query and apply them after computing
    let savedProjections: string[] | undefined;
    if (this.opts.computedExprs && this.opts.computedExprs.length > 0) {
      savedProjections = innerQuery.projections;
      innerQuery.projections = [];
    }

    // If multi-column sort, we handle sort + limit/offset
    let externalLimit: number | undefined;
    let externalOffset: number | undefined;
    if (hasMultiSort) {
      externalLimit = innerQuery.limit;
      externalOffset = innerQuery.offset;
      delete innerQuery.sortColumn;
      delete innerQuery.sortDirection;
      delete innerQuery.limit;
      delete innerQuery.offset;
    }

    const result = await this.inner.execute(innerQuery);
    let rows: Row[] = result.rows as Row[];

    // 1. Apply WHERE expression filter
    if (this.opts.whereExpr) {
      const whereExpr = this.opts.whereExpr;
      rows = rows.filter(row => isTruthy(evaluateExpr(whereExpr, row)));
    }

    // 2. Manual aggregation (when WHERE couldn't be pushed down + aggregates requested)
    if (manualAggregation && savedAggregates) {
      rows = aggregate(rows, savedAggregates, savedGroupBy);
    }

    // 3. Computed SELECT expressions (CASE, CAST, arithmetic)
    if (this.opts.computedExprs && this.opts.computedExprs.length > 0) {
      const exprs = this.opts.computedExprs;
      rows = rows.map(row => {
        const newRow: Row = { ...row };
        for (const { alias, expr } of exprs) {
          newRow[alias] = evaluateExpr(expr, row) as Row[string];
        }
        return newRow;
      });
    }

    // 3b. Apply deferred projections (after computed expressions added their columns)
    if (savedProjections && savedProjections.length > 0) {
      const keep = new Set([...savedProjections, ...((this.opts.computedExprs ?? []).map(e => e.alias))]);
      rows = rows.map(row => {
        const out: Row = {};
        for (const k of keep) if (k in row) out[k] = row[k];
        return out;
      });
    }

    // 4. HAVING filter (post-aggregation)
    if (this.opts.havingExpr) {
      const havingExpr = this.opts.havingExpr;
      rows = rows.filter(row => isTruthy(evaluateExpr(havingExpr, row)));
    }

    // 5. Multi-column ORDER BY
    if (hasMultiSort) {
      if (rows === (result.rows as Row[])) rows = [...rows];
      const orderBy = this.opts.allOrderBy!;
      rows.sort((a, b) => {
        for (const { column, direction } of orderBy) {
          const av = a[column], bv = b[column];
          if (av === bv) continue;
          if (av === null || av === undefined) return 1;
          if (bv === null || bv === undefined) return -1;
          let cmp: number;
          if (typeof av === "number" && typeof bv === "number") cmp = av - bv;
          else cmp = String(av).localeCompare(String(bv));
          if (cmp !== 0) return direction === "desc" ? -cmp : cmp;
        }
        return 0;
      });

      // Re-apply offset + limit
      if (externalOffset) rows = rows.slice(externalOffset);
      if (externalLimit !== undefined) rows = rows.slice(0, externalLimit);
    }

    return { ...result, rows, rowCount: rows.length };
  }

  async explain(query: QueryDescriptor): Promise<import("../types.js").ExplainResult> {
    return this.inner.explain(query);
  }
}

/** Manual aggregation for when WHERE couldn't be fully pushed down */
function aggregate(rows: Row[], aggregates: AggregateOp[], groupBy?: string[]): Row[] {
  if (!groupBy || groupBy.length === 0) {
    const result: Row = {};
    for (const agg of aggregates) {
      result[agg.alias ?? `${agg.fn}_${agg.column}`] = computeAgg(rows, agg);
    }
    return [result];
  }

  const groups = new Map<string, Row[]>();
  for (const row of rows) {
    let key = "";
    for (let g = 0; g < groupBy.length; g++) {
      if (g > 0) key += "\0";
      const v = row[groupBy[g]];
      key += v === null || v === undefined ? "\x01NULL\x01" : String(v);
    }
    let group = groups.get(key);
    if (!group) {
      group = [];
      groups.set(key, group);
    }
    group.push(row);
  }

  const results: Row[] = [];
  for (const group of groups.values()) {
    const row: Row = {};
    for (const col of groupBy) {
      row[col] = group[0][col];
    }
    for (const agg of aggregates) {
      row[agg.alias ?? `${agg.fn}_${agg.column}`] = computeAgg(group, agg);
    }
    results.push(row);
  }
  return results;
}

function computeAgg(rows: Row[], agg: AggregateOp): number | bigint | string | boolean | Float32Array | null {
  if (agg.fn === "count") {
    if (agg.column === "*") return rows.length;
    let cnt = 0;
    for (const r of rows) if (r[agg.column] != null) cnt++;
    return cnt;
  }

  if (agg.fn === "count_distinct") {
    const seen = new Set<string>();
    for (const row of rows) {
      const v = row[agg.column];
      if (v !== null && v !== undefined) seen.add(String(v));
    }
    return seen.size;
  }

  const needValues = agg.fn === "median" || agg.fn === "percentile" || agg.fn === "stddev" || agg.fn === "variance";
  const values: number[] = needValues ? [] : (undefined as unknown as number[]);
  let sum = 0, count = 0, min = Infinity, max = -Infinity;

  for (const row of rows) {
    const v = row[agg.column];
    if (v === null || v === undefined) continue;
    const num = typeof v === "number" ? v : typeof v === "bigint" ? Number(v) : parseFloat(String(v));
    if (isNaN(num)) continue;
    sum += num; count++;
    if (num < min) min = num;
    if (num > max) max = num;
    if (needValues) values.push(num);
  }

  if (count === 0) return null;

  switch (agg.fn) {
    case "sum": return sum;
    case "avg": return sum / count;
    case "min": return min;
    case "max": return max;
    case "stddev": {
      const mean = sum / count;
      let sq = 0;
      for (const v of values) sq += (v - mean) ** 2;
      return Math.sqrt(sq / count);
    }
    case "variance": {
      const mean = sum / count;
      let sq = 0;
      for (const v of values) sq += (v - mean) ** 2;
      return sq / count;
    }
    case "median": {
      values.sort((a, b) => a - b);
      const mid = Math.floor(count / 2);
      return count % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
    }
    case "percentile": {
      if (agg.percentileTarget === undefined) return null;
      values.sort((a, b) => a - b);
      const p = agg.percentileTarget;
      const idx = p * (count - 1);
      const lo = Math.floor(idx);
      const hi = Math.ceil(idx);
      return lo === hi ? values[lo] : values[lo] + (values[hi] - values[lo]) * (idx - lo);
    }
    default: return null;
  }
}
