import type {
  AggregateOp,
  AppendResult,
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
export class TableQuery {
  private _table: string;
  private _filters: FilterOp[] = [];
  private _projections: string[] = [];
  private _sortColumn?: string;
  private _sortDirection?: "asc" | "desc";
  private _limit?: number;
  private _vectorSearch?: VectorSearchParams;
  private _aggregates: AggregateOp[] = [];
  private _groupBy: string[] = [];
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
    this._limit = n;
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

  /** Append rows to this table. Uses CAS coordination for safe concurrent writes. */
  async append(rows: Record<string, unknown>[]): Promise<AppendResult> {
    if (!this._executor.append) {
      throw new Error("append() requires an executor with write support");
    }
    return this._executor.append(this._table, rows);
  }

  /** Execute the query. Resolves page ranges from cached footer, issues parallel Range reads. */
  async exec(): Promise<QueryResult> {
    return this._executor.execute(this.toDescriptor());
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
      vectorSearch: this._vectorSearch,
      aggregates: this._aggregates.length > 0 ? this._aggregates : undefined,
      groupBy: this._groupBy.length > 0 ? this._groupBy : undefined,
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
  vectorSearch?: VectorSearchParams;
  aggregates?: AggregateOp[];
  groupBy?: string[];
}

/** Interface for query execution backends (local, DO, browser) */
export interface QueryExecutor {
  execute(query: QueryDescriptor): Promise<QueryResult>;
  /** Optional: append rows (available on local and remote executors with write support) */
  append?(table: string, rows: Record<string, unknown>[]): Promise<AppendResult>;
  /** Optional: streaming execution (available on remote executors) */
  executeStream?(query: QueryDescriptor): Promise<ReadableStream<Row>>;
}
