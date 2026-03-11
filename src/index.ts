import { DataFrame, TableQuery } from "./client.js";
import type { QueryDescriptor, QueryExecutor } from "./client.js";
import type { AppendOptions, AppendResult, DropResult, ExplainResult, QueryResult, Row, QueryDORpc, MasterDORpc } from "./types.js";
import { LocalExecutor } from "./local-executor.js";
import { createFromJSON, createFromCSV, createDemo } from "./convenience.js";
import { sqlToDescriptor, buildSqlDataFrame } from "./sql/index.js";

export { MasterDO } from "./master-do.js";
export { QueryDO } from "./query-do.js";
export { FragmentDO } from "./fragment-do.js";
export { WorkerPool, R2Partitioner } from "./worker-pool.js";
export { WorkerDO } from "./worker-do.js";
export type { R2Partition, WorkerDORpc } from "./worker-pool.js";
export { ReaderRegistry, FileDataSource, UrlDataSource } from "./reader.js";
export type { FormatReader, DataSource } from "./reader.js";
export { DataFrame, TableQuery, MaterializedExecutor } from "./client.js";
export { LazyResultHandle } from "./client.js";
export { descriptorToCode } from "./descriptor-to-code.js";
export {
  FilterOperator, AggregateOperator, TopKOperator, ProjectOperator,
  ComputedColumnOperator, HashJoinOperator, ExternalSortOperator,
  InMemorySortOperator, WindowOperator, DistinctOperator, SetOperator,
  LimitOperator, SubqueryInOperator,
  drainPipeline, buildEdgePipeline, FsSpillBackend,
} from "./operators.js";
export type { Operator, RowBatch } from "./operators.js";
export { QueryModeError } from "./errors.js";
export type { ErrorCode } from "./errors.js";
export { LocalExecutor } from "./local-executor.js";
export { bigIntReplacer } from "./decode.js";
export { R2SpillBackend } from "./r2-spill.js";
export { createFromJSON, createFromCSV, createDemo } from "./convenience.js";
export { formatResultSummary, formatExplain, formatBytes } from "./format.js";
export type { LocalTimingInfo } from "./format.js";
export { sqlToDescriptor } from "./sql/index.js";
export { parse as parseSql } from "./sql/index.js";
export { SqlParseError, SqlLexerError } from "./sql/index.js";
export { HnswIndex, cosineDistance, l2DistanceSq, dotDistance } from "./hnsw.js";
export type { HnswOptions } from "./hnsw.js";
export { MaterializationCache, queryHashKey } from "./lazy.js";
export type { MaterializationCacheOptions } from "./lazy.js";
export type { QueryExecutor, QueryDescriptor, ProgressInfo, CollectOptions, PipeStage } from "./client.js";
export type {
  Env,
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
  JoinKeys,
  JoinType,
  WindowSpec,
  VectorOpts,
  QueryResult,
  Row,
  VectorSearchParams,
  IcebergSchema,
  IcebergDatasetMeta,
  AppendOptions,
  AppendResult,
  DropResult,
  ExplainResult,
  VectorIndexInfo,
  VersionInfo,
  DiffResult,
  QueryDORpc,
  MasterDORpc,
} from "./types.js";

/**
 * QueryMode — serverless columnar query engine.
 *
 * Usage:
 *   // Edge mode (Durable Objects + R2)
 *   const eq = QueryMode.remote(env.QUERY_DO, { region: "SJC" })
 *   const results = await eq.table("users").filter("age", "gt", 25).exec()
 *
 *   // Local mode (Node/Bun + filesystem)
 *   const eq = QueryMode.local()
 *   const results = await eq.table("./data/users.lance").select("name").exec()
 */
export class QueryMode {
  private executor: QueryExecutor;

  private constructor(executor: QueryExecutor) {
    this.executor = executor;
  }

  /**
   * Create an QueryMode client backed by a regional Query DO.
   * @param region - Datacenter code (e.g., "SJC", "NRT"). Must match worker.ts naming.
   *                 Defaults to "default" for direct SDK use.
   * @param locationHint - Cloudflare locationHint for DO placement.
   */
  static remote(
    queryDoNamespace: DurableObjectNamespace,
    options?: { region?: string; locationHint?: string; masterDoNamespace?: DurableObjectNamespace },
  ): QueryMode {
    const executor = new RemoteExecutor(
      queryDoNamespace,
      options?.region ?? "default",
      options?.locationHint,
      options?.masterDoNamespace,
    );
    return new QueryMode(executor);
  }

  /** Create an QueryMode client for local use (Node/Bun, reads files from disk or URLs). */
  static local(opts?: { wasmModule?: WebAssembly.Module; memoryBudgetBytes?: number }): QueryMode {
    const executor = opts ? new LocalExecutor(opts) : new LocalExecutor();
    return new QueryMode(executor);
  }

  /** Start building a query against a table. Returns a lazy DataFrame. */
  table(name: string): DataFrame {
    return new DataFrame(name, this.executor);
  }

  /** Parse a SQL query and return a DataFrame backed by the existing pipeline. */
  sql(query: string): DataFrame {
    return buildSqlDataFrame(query, this.executor);
  }

  /**
   * Execute a multi-table query with explicit orchestration.
   * Use this for JOINs — write the join logic in code, not SQL.
   */
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

  /** List all known tables with column names and row counts (edge mode only). */
  async tables(): Promise<{ name: string; columns: string[]; totalRows: number }[]> {
    if (!(this.executor instanceof RemoteExecutor)) {
      throw new Error("tables() is only available in edge mode. Use .table(path).describe() for local files.");
    }
    const result = await (this.executor as RemoteExecutor).listTables();
    return result.tables as { name: string; columns: string[]; totalRows: number }[];
  }
}

/**
 * Executor that calls regional Query DO via Worker RPC.
 * Zero-serialization: structured clone instead of JSON, no HTTP overhead.
 * RPC sessions survive DO hibernation — each session is one billable request.
 */
class RemoteExecutor implements QueryExecutor {
  private namespace: DurableObjectNamespace;
  private masterNamespace?: DurableObjectNamespace;
  private region: string;
  private locationHint?: string;

  constructor(namespace: DurableObjectNamespace, region: string, locationHint?: string, masterNamespace?: DurableObjectNamespace) {
    this.namespace = namespace;
    this.region = region;
    this.locationHint = locationHint;
    this.masterNamespace = masterNamespace;
  }

  /** Get a typed RPC handle for the regional Query DO. */
  private getQueryHandle(): QueryDORpc {
    const doName = `query-${this.region}`;
    const id = this.namespace.idFromName(doName);
    const doRef = this.locationHint
      ? this.namespace.get(id, { locationHint: this.locationHint as DurableObjectLocationHint })
      : this.namespace.get(id);
    return doRef as unknown as QueryDORpc;
  }

  /** Execute query via RPC — zero JSON serialization overhead. */
  async execute(query: QueryDescriptor): Promise<QueryResult> {
    const rpc = this.getQueryHandle();
    return rpc.queryRpc(query);
  }

  /** Append rows via RPC to Master DO. Partitioned writes fan out to separate MasterDOs. */
  async append(table: string, rows: Record<string, unknown>[], options?: AppendOptions): Promise<AppendResult> {
    if (!this.masterNamespace) {
      throw new Error("append() requires masterDoNamespace — pass it via QueryMode.remote(queryDO, { masterDO })");
    }

    // Partitioned writes: split by partition key, route each group to a different MasterDO
    if (options?.partitionBy) {
      const partCol = options.partitionBy;
      const groups = new Map<string, Record<string, unknown>[]>();
      for (const row of rows) {
        const key = row[partCol] === null || row[partCol] === undefined ? "__null__" : String(row[partCol]);
        let group = groups.get(key);
        if (!group) { group = []; groups.set(key, group); }
        group.push(row);
      }

      // Fan out: each partition group goes to a MasterDO named by table+partition
      const results = await Promise.all(
        [...groups.entries()].map(([partValue, groupRows]) => {
          const doName = `master-${table}-${partValue}`;
          const id = this.masterNamespace!.idFromName(doName);
          const rpc = this.masterNamespace!.get(id) as unknown as MasterDORpc;
          return rpc.appendRpc(table, groupRows, options);
        }),
      );

      const totalWritten = results.reduce((s, r) => s + r.rowsWritten, 0);
      return {
        ...results[results.length - 1],
        rowsWritten: totalWritten,
        metadata: { ...options.metadata, partitionBy: partCol, partitions: String(groups.size) },
      };
    }

    const id = this.masterNamespace.idFromName("master");
    const masterRpc = this.masterNamespace.get(id) as unknown as MasterDORpc;
    return masterRpc.appendRpc(table, rows, options);
  }

  /** Drop a table via RPC to Master DO — deletes all R2 objects and metadata. */
  async drop(table: string): Promise<DropResult> {
    if (!this.masterNamespace) {
      throw new Error("drop() requires masterDoNamespace — pass it via QueryMode.remote(queryDO, { masterDO })");
    }
    const id = this.masterNamespace.idFromName("master");
    const masterRpc = this.masterNamespace.get(id) as unknown as MasterDORpc;
    return masterRpc.dropRpc(table);
  }

  /** Stream query results via RPC — columnar binary framed stream. */
  async executeStream(query: QueryDescriptor): Promise<ReadableStream<Row>> {
    const { decodeColumnarRun } = await import("./r2-spill.js");
    const rpc = this.getQueryHandle();
    const byteStream = await rpc.streamRpc(query);

    // Parse length-prefixed columnar binary frames into Row objects
    return new ReadableStream<Row>({
      async start(controller) {
        const reader = byteStream.getReader();
        let pending: Uint8Array = new Uint8Array(0);

        const concat = (a: Uint8Array, b: Uint8Array): Uint8Array => {
          const out = new Uint8Array(a.length + b.length);
          out.set(a, 0);
          out.set(b, a.length);
          return out;
        };

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            pending = pending.length > 0 ? concat(pending, value) : value;

            // Process complete frames
            while (pending.length >= 4) {
              const frameLen = new DataView(pending.buffer as ArrayBuffer, pending.byteOffset).getUint32(0, true);
              if (pending.length < 4 + frameLen) break; // wait for more data

              const frameBuf = (pending.buffer as ArrayBuffer).slice(
                pending.byteOffset + 4,
                pending.byteOffset + 4 + frameLen,
              );
              pending = pending.subarray(4 + frameLen);

              for (const row of decodeColumnarRun(frameBuf)) {
                controller.enqueue(row);
              }
            }
          }
          controller.close();
        } catch (err) {
          controller.error(err);
        }
      },
    });
  }

  /** Count via RPC — returns number directly, no JSON wrapper. */
  async count(query: QueryDescriptor): Promise<number> {
    const rpc = this.getQueryHandle();
    return rpc.countRpc(query);
  }

  /** Exists via RPC — returns boolean directly. */
  async exists(query: QueryDescriptor): Promise<boolean> {
    const rpc = this.getQueryHandle();
    return rpc.existsRpc(query);
  }

  /** First via RPC — returns Row directly. */
  async first(query: QueryDescriptor): Promise<Row | null> {
    const rpc = this.getQueryHandle();
    return rpc.firstRpc(query);
  }

  /** Explain via RPC — returns ExplainResult directly. */
  async explain(query: QueryDescriptor): Promise<ExplainResult> {
    const rpc = this.getQueryHandle();
    return rpc.explainRpc(query);
  }

  /** List all tables known to the regional Query DO. */
  async listTables(): Promise<{ tables: unknown[] }> {
    const rpc = this.getQueryHandle();
    return rpc.listTablesRpc();
  }
}
