/** SQL frontend — parse SQL strings into QueryDescriptor for the operator pipeline */

export { parse } from "./parser.js";
export { compile, compileFull } from "./compiler.js";
export type { SqlCompileResult } from "./compiler.js";
export { tokenize, SqlLexerError } from "./lexer.js";
export { SqlParseError } from "./parser.js";
export { evaluateExpr, isTruthy } from "./evaluator.js";
export { SqlWrappingExecutor } from "./executor.js";
export type { SqlExecutorOpts } from "./executor.js";
export type { Token, TokenType } from "./lexer.js";
export type { SelectStmt, SqlExpr, TableRef, SelectItem, SqlOrderBy, SqlGroupBy, CteDef } from "./ast.js";
export type { CompiledCte } from "./compiler.js";

import type { QueryDescriptor, QueryExecutor } from "../client.js";
import { DataFrame } from "../client.js";
import { parse } from "./parser.js";
import { compile, compileFull } from "./compiler.js";
import { SqlWrappingExecutor } from "./executor.js";

export function sqlToDescriptor(sql: string): QueryDescriptor {
  const stmt = parse(sql);
  return compile(stmt);
}

/** Build a DataFrame from SQL, wiring up the wrapping executor for features beyond FilterOp[]. */
export function buildSqlDataFrame(sql: string, executor: QueryExecutor): DataFrame {
  const stmt = parse(sql);
  const result = compileFull(stmt);
  let desc = result.descriptor;

  // Resolve CTEs: if main query's FROM references a CTE, inline its descriptor
  if (result.ctes && result.ctes.length > 0) {
    const cteMap = new Map(result.ctes.map(c => [c.name, c.result]));
    const cte = cteMap.get(desc.table);
    if (cte) {
      // Merge: CTE provides the base table + filters, main query adds its own
      const base = cte.descriptor;
      desc = {
        ...desc,
        table: base.table,
        filters: [...base.filters, ...desc.filters],
        filterGroups: [...(base.filterGroups ?? []), ...(desc.filterGroups ?? [])],
      };
    }
  }

  // Wrap executor if SQL uses features that can't be pushed down to FilterOp[]
  const needsWrapper = result.whereExpr || result.havingExpr || result.allOrderBy || result.computedExprs;
  const effectiveExecutor = needsWrapper
    ? new SqlWrappingExecutor(executor, {
        whereExpr: result.whereExpr,
        havingExpr: result.havingExpr,
        allOrderBy: result.allOrderBy,
        computedExprs: result.computedExprs,
      })
    : executor;

  return new DataFrame(desc.table, effectiveExecutor, {
    filters: desc.filters,
    filterGroups: desc.filterGroups ?? [],
    projections: desc.projections,
    sortColumn: desc.sortColumn,
    sortDirection: desc.sortDirection,
    limit: desc.limit,
    offset: desc.offset,
    aggregates: desc.aggregates ?? [],
    groupBy: desc.groupBy ?? [],
    vectorSearch: desc.vectorSearch,
    join: desc.join,
    windows: desc.windows ?? [],
    distinct: desc.distinct,
    setOperation: desc.setOperation,
    computedColumns: [],
    subqueryIn: [],
    deferredSubqueries: [],
  });
}
