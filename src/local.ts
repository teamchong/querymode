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

import { DataFrame, TableQuery } from "./client.js";
import type { QueryExecutor } from "./client.js";
import { LocalExecutor } from "./local-executor.js";
import { createFromJSON, createFromCSV, createDemo } from "./convenience.js";

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

  /** Create a DataFrame from an in-memory array of objects. No files needed. */
  static fromJSON<T extends Record<string, unknown>>(data: T[], tableName?: string): DataFrame {
    return createFromJSON(data, tableName);
  }

  /** Create a DataFrame from a CSV string. Auto-detects delimiter and infers types. */
  static fromCSV(csv: string, tableName?: string): Promise<DataFrame> {
    return createFromCSV(csv, tableName);
  }

  /** Create a demo DataFrame with 1000 rows of sample data. No files needed. */
  static demo(tableName?: string): DataFrame {
    return createDemo(tableName);
  }
}

export { LocalExecutor } from "./local-executor.js";
export { DataFrame, TableQuery, MaterializedExecutor } from "./client.js";
export { QueryModeError } from "./errors.js";
export { bigIntReplacer } from "./decode.js";
export { createFromJSON, createFromCSV, createDemo } from "./convenience.js";
export { formatResultSummary, formatExplain, formatBytes } from "./format.js";
export type { LocalTimingInfo } from "./format.js";
export type { QueryExecutor, QueryDescriptor, ProgressInfo, CollectOptions } from "./client.js";
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
