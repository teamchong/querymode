import { DataFrame, TableQuery } from "./client.js";
import type { QueryDescriptor, QueryExecutor } from "./client.js";
import type { AppendResult, ExplainResult, QueryResult, Row, QueryDORpc, MasterDORpc } from "./types.js";
import { LocalExecutor } from "./local-executor.js";

export { MasterDO } from "./master-do.js";
export { QueryDO } from "./query-do.js";
export { FragmentDO } from "./fragment-do.js";
export { WorkerPool, R2Partitioner } from "./worker-pool.js";
export { WorkerDO } from "./worker-do.js";
export type { R2Partition, WorkerDORpc } from "./worker-pool.js";
export { ReaderRegistry, FileDataSource, UrlDataSource } from "./reader.js";
export type { FormatReader, DataSource } from "./reader.js";
export { DataFrame, TableQuery } from "./client.js";
export { LazyResultHandle } from "./client.js";
export { QueryModeError } from "./errors.js";
export { LocalExecutor } from "./local-executor.js";
export { bigIntReplacer } from "./decode.js";
export { HnswIndex, cosineDistance, l2DistanceSq, dotDistance } from "./hnsw.js";
export type { HnswOptions } from "./hnsw.js";
export { MaterializationCache, queryHashKey } from "./lazy.js";
export type { MaterializationCacheOptions } from "./lazy.js";
export type { QueryExecutor, QueryDescriptor } from "./client.js";
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
  AppendResult,
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

  /**
   * Execute a multi-table query with explicit orchestration.
   * Use this for JOINs — write the join logic in code, not SQL.
   */
  async query<T>(fn: () => Promise<T>): Promise<T> {
    return fn();
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

  /** Append rows via RPC to Master DO. */
  async append(table: string, rows: Record<string, unknown>[]): Promise<AppendResult> {
    if (!this.masterNamespace) {
      throw new Error("append() requires masterDoNamespace — pass it via QueryMode.remote(queryDO, { masterDO })");
    }
    const id = this.masterNamespace.idFromName("master");
    const masterRpc = this.masterNamespace.get(id) as unknown as MasterDORpc;
    return masterRpc.appendRpc(table, rows);
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
        let pending = new Uint8Array(0);

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
            const chunk = new Uint8Array(value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength)) as Uint8Array<ArrayBuffer>;
            pending = pending.length > 0 ? concat(pending, chunk) : chunk;

            // Process complete frames
            while (pending.length >= 4) {
              const frameLen = new DataView(pending.buffer, pending.byteOffset).getUint32(0, true);
              if (pending.length < 4 + frameLen) break; // wait for more data

              const frameBuf = pending.buffer.slice(
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
}
