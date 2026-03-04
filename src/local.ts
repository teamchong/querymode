/**
 * Node-safe entry point for querymode.
 *
 * This module re-exports only the parts of querymode that work in Node/Bun
 * without any Cloudflare Durable Object dependencies (query-do, master-do,
 * fragment-do, wasm-module).
 *
 * Usage:
 *   import { QueryMode, LocalExecutor, TableQuery } from "querymode/local";
 *   const qm = QueryMode.local();
 *   const result = await qm.table("./data/users.lance").select("name").exec();
 */

import { TableQuery } from "./client.js";
import type { QueryExecutor } from "./client.js";
import { LocalExecutor } from "./local-executor.js";

/**
 * QueryMode — local-only entry point (no Cloudflare DO dependencies).
 */
export class QueryMode {
  private executor: QueryExecutor;

  private constructor(executor: QueryExecutor) {
    this.executor = executor;
  }

  /** Create a QueryMode client for local use (Node/Bun, reads files from disk or URLs). */
  static local(opts?: { wasmModule?: WebAssembly.Module; memoryBudgetBytes?: number }): QueryMode {
    const executor = opts ? new LocalExecutor(opts) : new LocalExecutor();
    return new QueryMode(executor);
  }

  /** Start building a query against a table. */
  table(name: string): TableQuery {
    return new TableQuery(name, this.executor);
  }

  /** Execute a multi-table query with explicit orchestration. */
  async query<T>(fn: () => Promise<T>): Promise<T> {
    return fn();
  }
}

export { LocalExecutor } from "./local-executor.js";
export { TableQuery } from "./client.js";
export { QueryModeError } from "./errors.js";
export { bigIntReplacer } from "./decode.js";
export type { QueryExecutor, QueryDescriptor } from "./client.js";
export type {
  Footer,
  TableMeta,
  ColumnMeta,
  PageInfo,
  PageEncoding,
  DataType,
  DatasetMeta,
  ManifestInfo,
  FragmentInfo,
  FilterOp,
  AggregateOp,
  JoinDescriptor,
  QueryResult,
  Row,
  VectorSearchParams,
  IcebergSchema,
  IcebergDatasetMeta,
  AppendResult,
  ExplainResult,
  VectorIndexInfo,
} from "./types.js";
