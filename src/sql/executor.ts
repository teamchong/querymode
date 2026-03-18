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
import { computePartialAgg, finalizePartialAgg } from "../partial-agg.js";

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

    // Strip projections when we need columns not in SELECT:
    // - computedExprs reference arbitrary columns for CASE/CAST/arithmetic
    // - havingExpr references aggregate columns that may not be in SELECT
    // - multi-column ORDER BY may sort on columns not in SELECT
    // Projections are re-applied after HAVING and ORDER BY (step 5b).
    let savedProjections: string[] | undefined;
    if ((this.opts.computedExprs && this.opts.computedExprs.length > 0) || this.opts.havingExpr || hasMultiSort) {
      savedProjections = innerQuery.projections;
      innerQuery.projections = [];
    }

    // Strip limit/offset when we post-filter (HAVING, WHERE) or multi-sort — otherwise
    // the inner executor limits rows before our filter removes some, giving fewer results.
    let externalLimit: number | undefined;
    let externalOffset: number | undefined;
    const needsExternalLimit = hasMultiSort || !!this.opts.havingExpr || hasWhereExpr || manualAggregation;
    if (needsExternalLimit) {
      externalLimit = innerQuery.limit;
      externalOffset = innerQuery.offset;
      delete innerQuery.limit;
      delete innerQuery.offset;
    }
    if (hasMultiSort) {
      delete innerQuery.sortColumn;
      delete innerQuery.sortDirection;
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
          else if (typeof av === "bigint" && typeof bv === "bigint") cmp = av < bv ? -1 : av > bv ? 1 : 0;
          else if (typeof av === "bigint" && typeof bv === "number") { const bb = BigInt(Math.trunc(bv)); cmp = av < bb ? -1 : av > bb ? 1 : 0; }
          else if (typeof av === "number" && typeof bv === "bigint") { const ab = BigInt(Math.trunc(av)); cmp = ab < bv ? -1 : ab > bv ? 1 : 0; }
          else cmp = String(av).localeCompare(String(bv));
          if (cmp !== 0) return direction === "desc" ? -cmp : cmp;
        }
        return 0;
      });
    }

    // 5b. Apply deferred projections (after HAVING/ORDER BY which may reference non-SELECT columns)
    if (savedProjections && savedProjections.length > 0) {
      const keep = new Set([...savedProjections, ...((this.opts.computedExprs ?? []).map(e => e.alias))]);
      rows = rows.map(row => {
        const out: Row = {};
        for (const k of keep) if (k in row) out[k] = row[k];
        return out;
      });
    }

    // 6. Re-apply offset + limit (stripped when post-filtering or multi-sorting)
    if (needsExternalLimit) {
      if (externalOffset) rows = rows.slice(externalOffset);
      if (externalLimit !== undefined) rows = rows.slice(0, externalLimit);
    }

    const finalColumns = savedProjections && savedProjections.length > 0
      ? [...new Set([...savedProjections, ...((this.opts.computedExprs ?? []).map(e => e.alias))])]
      : result.columns;
    return { ...result, rows, rowCount: rows.length, columns: finalColumns };
  }

  async explain(query: QueryDescriptor): Promise<import("../types.js").ExplainResult> {
    return this.inner.explain(query);
  }
}

/** Manual aggregation for when WHERE couldn't be fully pushed down */
function aggregate(rows: Row[], aggregates: AggregateOp[], groupBy?: string[]): Row[] {
  const query = { table: "", filters: [], projections: [], aggregates, groupBy } as QueryDescriptor;
  const partial = computePartialAgg(rows, query);
  return finalizePartialAgg(partial, query);
}
