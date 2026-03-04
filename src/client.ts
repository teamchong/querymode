import type {
  AggregateOp,
  AppendResult,
  ExplainResult,
  FilterOp,
  QueryResult,
  Row,
  VectorSearchParams,
} from "./types.js";

/**
 * Code-first query builder. No SQL — just TypeScript.
 *
 * Usage:
 *   const results = await querymode
 *     .table("users")
 *     .filter(r => r.age > 25)
 *     .select("name", "email")
 *     .sort("age", "desc")
 *     .limit(100)
 *     .exec()
 */
export class TableQuery<T extends Row = Row> {
  private _table: string;
  private _filters: FilterOp[] = [];
  private _projections: string[] = [];
  private _sortColumn?: string;
  private _sortDirection?: "asc" | "desc";
  private _limit?: number;
  private _offset?: number;
  private _vectorSearch?: VectorSearchParams;
  private _aggregates: AggregateOp[] = [];
  private _groupBy: string[] = [];
  private _cacheTTL?: number;
  private _executor: QueryExecutor;

  constructor(table: string, executor: QueryExecutor) {
    this._table = table;
    this._executor = executor;
  }

  /** Filter rows by a column predicate. Pushes down to page-level skipping. */
  where(column: string, op: FilterOp["op"], value: FilterOp["value"]): this {
    this._filters.push({ column, op, value });
    return this;
  }

  /** Shorthand: .filter("age", "gt", 25) */
  filter(column: string, op: FilterOp["op"], value: FilterOp["value"]): this {
    return this.where(column, op, value);
  }

  /** Select specific columns. Only these byte ranges are fetched from R2. */
  select(...columns: string[]): this {
    this._projections = columns;
    return this;
  }

  /** Sort results by a column. With .limit(), uses a top-K heap (O(K) memory). */
  sort(column: string, direction: "asc" | "desc" = "asc"): this {
    this._sortColumn = column;
    this._sortDirection = direction;
    return this;
  }

  /** Limit the number of returned rows. Enables early termination. */
  limit(n: number): this {
    if (n < 0) throw new Error("limit() must be non-negative");
    this._limit = n;
    return this;
  }

  /** Skip the first N rows. Enables offset-based pagination. */
  offset(n: number): this {
    this._offset = n;
    return this;
  }

  /** Keyset pagination: fetch rows after the given cursor value. Requires sort(). */
  after(value: FilterOp["value"]): this {
    if (!this._sortColumn) throw new Error("after() requires sort()");
    const op = this._sortDirection === "desc" ? "lt" : "gt";
    this._filters.push({ column: this._sortColumn, op, value });
    return this;
  }

  /** Aggregate a column: .aggregate("sum", "amount") or .aggregate("count", "*") */
  aggregate(fn: AggregateOp["fn"], column: string, alias?: string): this {
    this._aggregates.push({ fn, column, alias });
    return this;
  }

  /** Group results by one or more columns. Used with .aggregate(). */
  groupBy(...columns: string[]): this {
    this._groupBy = columns;
    return this;
  }

  /** SIMD vector similarity search on a fixed_size_list column. */
  vector(column: string, queryVector: Float32Array, topK: number): this {
    this._vectorSearch = { column, queryVector, topK };
    return this;
  }

  /** Enable query result caching with a TTL in milliseconds. Remote executor only. */
  cache(opts: { ttl: number }): this {
    this._cacheTTL = opts.ttl;
    return this;
  }

  /** Return the count of matching rows without full materialization. */
  async count(): Promise<number> {
    if (this._executor.count) return this._executor.count(this.toDescriptor());
    const desc = this.toDescriptor();
    desc.aggregates = [{ fn: "count", column: "*" }];
    const result = await this._executor.execute(desc);
    return (result.rows[0]?.["count_*"] as number) ?? 0;
  }

  /** Return true if at least one row matches. */
  async exists(): Promise<boolean> {
    if (this._executor.exists) return this._executor.exists(this.toDescriptor());
    const desc = this.toDescriptor();
    desc.limit = 1;
    if (desc.projections.length === 0) desc.projections = [];
    const result = await this._executor.execute(desc);
    return result.rowCount > 0;
  }

  /** Return the first matching row, or null. */
  async first(): Promise<T | null> {
    if (this._executor.first) return this._executor.first(this.toDescriptor()) as Promise<T | null>;
    const desc = this.toDescriptor();
    desc.limit = 1;
    const result = await this._executor.execute(desc);
    return (result.rows[0] as T) ?? null;
  }

  /** Inspect the query plan without executing. No data I/O is performed. */
  async explain(): Promise<ExplainResult> {
    if (!this._executor.explain) {
      throw new Error("explain() requires an executor with plan inspection support");
    }
    return this._executor.explain(this.toDescriptor());
  }

  /** Iterate over results in batches. Processes pages lazily — stops when consumer breaks. */
  cursor(opts?: { batchSize?: number }): AsyncIterable<Row[]> {
    if (!this._executor.cursor) {
      throw new Error("cursor() requires an executor with cursor support");
    }
    return this._executor.cursor(this.toDescriptor(), opts?.batchSize ?? 1000);
  }

  /** Append rows to this table. Uses CAS coordination for safe concurrent writes. */
  async append(rows: Record<string, unknown>[]): Promise<AppendResult> {
    if (!this._executor.append) {
      throw new Error("append() requires an executor with write support");
    }
    return this._executor.append(this._table, rows);
  }

  /** Execute the query. Resolves page ranges from cached footer, issues parallel Range reads. */
  async exec(): Promise<QueryResult<T>> {
    return this._executor.execute(this.toDescriptor()) as Promise<QueryResult<T>>;
  }

  /** Execute and return an NDJSON stream of rows. Only works with RemoteExecutor. */
  async execStream(): Promise<ReadableStream<Row>> {
    if (!this._executor.executeStream) {
      throw new Error("execStream() requires a remote executor with streaming support");
    }
    return this._executor.executeStream(this.toDescriptor());
  }

  private toDescriptor(): QueryDescriptor {
    return {
      table: this._table,
      filters: this._filters,
      projections: this._projections,
      sortColumn: this._sortColumn,
      sortDirection: this._sortDirection,
      limit: this._limit,
      offset: this._offset,
      vectorSearch: this._vectorSearch,
      aggregates: this._aggregates.length > 0 ? this._aggregates : undefined,
      groupBy: this._groupBy.length > 0 ? this._groupBy : undefined,
      cacheTTL: this._cacheTTL,
    };
  }
}

/** Internal query descriptor passed to the executor */
export interface QueryDescriptor {
  table: string;
  filters: FilterOp[];
  projections: string[];
  sortColumn?: string;
  sortDirection?: "asc" | "desc";
  limit?: number;
  offset?: number;
  vectorSearch?: VectorSearchParams;
  aggregates?: AggregateOp[];
  groupBy?: string[];
  cacheTTL?: number;
}

/** Interface for query execution backends (local, DO, browser) */
export interface QueryExecutor {
  execute(query: QueryDescriptor): Promise<QueryResult>;
  /** Optional: append rows (available on local and remote executors with write support) */
  append?(table: string, rows: Record<string, unknown>[]): Promise<AppendResult>;
  /** Optional: streaming execution (available on remote executors) */
  executeStream?(query: QueryDescriptor): Promise<ReadableStream<Row>>;
  /** Optional: count without full materialization */
  count?(query: QueryDescriptor): Promise<number>;
  /** Optional: existence check with limit(1) */
  exists?(query: QueryDescriptor): Promise<boolean>;
  /** Optional: return first matching row */
  first?(query: QueryDescriptor): Promise<Row | null>;
  /** Optional: plan inspection without execution */
  explain?(query: QueryDescriptor): Promise<ExplainResult>;
  /** Optional: lazy batch iteration */
  cursor?(query: QueryDescriptor, batchSize: number): AsyncIterable<Row[]>;
}
