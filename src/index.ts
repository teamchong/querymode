import { TableQuery } from "./client.js";
import type { QueryDescriptor, QueryExecutor } from "./client.js";
import type { AppendResult, ExplainResult, QueryResult, Row } from "./types.js";
import { bigIntReplacer } from "./decode.js";
import { LocalExecutor } from "./local-executor.js";

export { MasterDO } from "./master-do.js";
export { QueryDO } from "./query-do.js";
export { FragmentDO } from "./fragment-do.js";
export { TableQuery } from "./client.js";
export { QueryModeError } from "./errors.js";
export { LocalExecutor } from "./local-executor.js";
export { bigIntReplacer } from "./decode.js";
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
  QueryResult,
  Row,
  VectorSearchParams,
  IcebergSchema,
  IcebergDatasetMeta,
  AppendResult,
  ExplainResult,
  VectorIndexInfo,
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
  static local(wasmModule?: WebAssembly.Module): QueryMode {
    const executor = new LocalExecutor(wasmModule);
    return new QueryMode(executor);
  }

  /** Start building a query against a table. */
  table(name: string): TableQuery {
    return new TableQuery(name, this.executor);
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
 * Executor that sends queries to a regional Query DO.
 * The DO has cached footers — no metadata round-trip needed.
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

  private getQueryDo() {
    const doName = `query-${this.region}`;
    const id = this.namespace.idFromName(doName);
    return this.locationHint
      ? this.namespace.get(id, { locationHint: this.locationHint as DurableObjectLocationHint })
      : this.namespace.get(id);
  }

  async execute(query: QueryDescriptor): Promise<QueryResult> {
    const queryDo = this.getQueryDo();

    const response = await queryDo.fetch(new Request("http://internal/query", {
      method: "POST",
      body: JSON.stringify(query, bigIntReplacer),
      headers: { "content-type": "application/json" },
    }));

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`QueryMode query failed: ${error}`);
    }

    return response.json() as Promise<QueryResult>;
  }

  async append(table: string, rows: Record<string, unknown>[]): Promise<AppendResult> {
    if (!this.masterNamespace) {
      throw new Error("append() requires masterDoNamespace — pass it via QueryMode.remote(queryDO, { masterDO })");
    }
    // Append goes to Master DO (single writer)
    const id = this.masterNamespace.idFromName("master");
    const masterDo = this.masterNamespace.get(id);

    const response = await masterDo.fetch(new Request("http://internal/append", {
      method: "POST",
      body: JSON.stringify({ table, rows }, bigIntReplacer),
      headers: { "content-type": "application/json" },
    }));

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`QueryMode append failed: ${error}`);
    }

    return response.json() as Promise<AppendResult>;
  }

  async executeStream(query: QueryDescriptor): Promise<ReadableStream<Row>> {
    const queryDo = this.getQueryDo();

    const response = await queryDo.fetch(new Request("http://internal/query/stream", {
      method: "POST",
      body: JSON.stringify(query, bigIntReplacer),
      headers: { "content-type": "application/json" },
    }));

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`QueryMode stream failed: ${error}`);
    }

    if (!response.body) throw new Error("No response body for stream");

    const decoder = new TextDecoder();
    const MAX_STREAM_BUFFER = 10 * 1024 * 1024; // 10MB
    let buffer = "";

    return new ReadableStream<Row>({
      async start(controller) {
        const reader = response.body!.getReader();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            if (buffer.length > MAX_STREAM_BUFFER) {
              controller.error(new Error("Stream buffer exceeded 10MB"));
              reader.cancel();
              return;
            }
            const lines = buffer.split("\n");
            buffer = lines.pop()!;
            for (const line of lines) {
              if (line.trim()) controller.enqueue(JSON.parse(line) as Row);
            }
          }
          if (buffer.trim()) controller.enqueue(JSON.parse(buffer) as Row);
          controller.close();
        } catch (err) {
          controller.error(err);
        }
      },
    });
  }

  private async postQuery<T>(path: string, query: QueryDescriptor): Promise<T> {
    const queryDo = this.getQueryDo();
    const response = await queryDo.fetch(new Request(`http://internal${path}`, {
      method: "POST",
      body: JSON.stringify(query, bigIntReplacer),
      headers: { "content-type": "application/json" },
    }));
    if (!response.ok) {
      const error = await response.text();
      throw new Error(`QueryMode ${path} failed: ${error}`);
    }
    return response.json() as Promise<T>;
  }

  async count(query: QueryDescriptor): Promise<number> {
    const body = await this.postQuery<{ count: number }>("/query/count", query);
    return body.count;
  }

  async exists(query: QueryDescriptor): Promise<boolean> {
    const body = await this.postQuery<{ exists: boolean }>("/query/exists", query);
    return body.exists;
  }

  async first(query: QueryDescriptor): Promise<Row | null> {
    const body = await this.postQuery<{ row: Row | null }>("/query/first", query);
    return body.row;
  }

  async explain(query: QueryDescriptor): Promise<ExplainResult> {
    return this.postQuery<ExplainResult>("/query/explain", query);
  }
}
