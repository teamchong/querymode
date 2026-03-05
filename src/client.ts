import type {
  AggregateOp,
  AppendResult,
  ExplainResult,
  FilterOp,
  JoinDescriptor,
  JoinKeys,
  JoinType,
  QueryResult,
  Row,
  VectorOpts,
  VectorSearchParams,
  WindowSpec,
} from "./types.js";

// ---------------------------------------------------------------------------
// Computed column definition (callback, not serializable)
// ---------------------------------------------------------------------------

interface ComputedColumnDef {
  alias: string;
  fn: (row: Row) => unknown;
}

// ---------------------------------------------------------------------------
// SubqueryIn definition (pre-resolved value set)
// ---------------------------------------------------------------------------

interface SubqueryInDef {
  column: string;
  valueSet: Set<string>;
}

// ---------------------------------------------------------------------------
// Deferred subquery — resolved at collect() time
// ---------------------------------------------------------------------------

interface DeferredSubquery {
  column: string;
  subquery: QueryDescriptor;
}

/**
 * Lazy DataFrame — code-first query builder. No SQL, just TypeScript.
 *
 * All methods return a **new** DataFrame (immutable). Nothing executes
 * until a terminal method is called: .collect(), .exec(), .first(), etc.
 *
 * Usage:
 *   const results = await qm
 *     .table("users")
 *     .filter("age", "gt", 25)
 *     .select("name", "email")
 *     .sort("age", "desc")
 *     .limit(100)
 *     .collect()
 */
export class DataFrame<T extends Row = Row> {
  private _table: string;
  private _filters: FilterOp[];
  private _projections: string[];
  private _sortColumn?: string;
  private _sortDirection?: "asc" | "desc";
  private _limit?: number;
  private _offset?: number;
  private _vectorSearch?: VectorSearchParams;
  private _aggregates: AggregateOp[];
  private _groupBy: string[];
  private _cacheTTL?: number;
  private _join?: JoinDescriptor;
  private _computedColumns: ComputedColumnDef[];
  private _windows: WindowSpec[];
  private _distinct?: string[];
  private _setOperation?: { mode: "union" | "union_all" | "intersect" | "except"; right: QueryDescriptor };
  private _subqueryIn: SubqueryInDef[];
  private _deferredSubqueries: DeferredSubquery[];
  private _version?: number;
  private _vectorEncoder?: (text: string) => Promise<Float32Array>;
  private _vectorQueryText?: string;
  private _executor: QueryExecutor;

  constructor(table: string, executor: QueryExecutor, init?: Partial<DataFrameInit>) {
    this._table = table;
    this._executor = executor;
    this._filters = init?.filters ?? [];
    this._projections = init?.projections ?? [];
    this._sortColumn = init?.sortColumn;
    this._sortDirection = init?.sortDirection;
    this._limit = init?.limit;
    this._offset = init?.offset;
    this._vectorSearch = init?.vectorSearch;
    this._aggregates = init?.aggregates ?? [];
    this._groupBy = init?.groupBy ?? [];
    this._cacheTTL = init?.cacheTTL;
    this._join = init?.join;
    this._computedColumns = init?.computedColumns ?? [];
    this._windows = init?.windows ?? [];
    this._distinct = init?.distinct;
    this._setOperation = init?.setOperation;
    this._subqueryIn = init?.subqueryIn ?? [];
    this._deferredSubqueries = init?.deferredSubqueries ?? [];
    this._version = init?.version;
    this._vectorEncoder = init?.vectorEncoder;
    this._vectorQueryText = init?.vectorQueryText;
  }

  /** Create a shallow clone with overrides — returns a new immutable DataFrame. */
  private derive(overrides: Partial<DataFrameInit>): DataFrame<T> {
    return new DataFrame<T>(this._table, this._executor, {
      filters: this._filters,
      projections: this._projections,
      sortColumn: this._sortColumn,
      sortDirection: this._sortDirection,
      limit: this._limit,
      offset: this._offset,
      vectorSearch: this._vectorSearch,
      aggregates: this._aggregates,
      groupBy: this._groupBy,
      cacheTTL: this._cacheTTL,
      join: this._join,
      computedColumns: this._computedColumns,
      windows: this._windows,
      distinct: this._distinct,
      setOperation: this._setOperation,
      subqueryIn: this._subqueryIn,
      deferredSubqueries: this._deferredSubqueries,
      version: this._version,
      vectorEncoder: this._vectorEncoder,
      vectorQueryText: this._vectorQueryText,
      ...overrides,
    });
  }

  // --- Plan builders (return new DataFrame, no execution) ---

  /** Filter rows by a column predicate. Pushes down to page-level skipping. */
  where(column: string, op: FilterOp["op"], value: FilterOp["value"]): DataFrame<T> {
    return this.derive({ filters: [...this._filters, { column, op, value }] });
  }

  /** Shorthand: .filter("age", "gt", 25) */
  filter(column: string, op: FilterOp["op"], value: FilterOp["value"]): DataFrame<T> {
    return this.where(column, op, value);
  }

  /** Select specific columns. Only these byte ranges are fetched from R2. */
  select(...columns: string[]): DataFrame {
    return this.derive({ projections: columns }) as DataFrame;
  }

  /** Sort results by a column. With .limit(), uses a top-K heap (O(K) memory). */
  sort(column: string, direction: "asc" | "desc" = "asc"): DataFrame<T> {
    return this.derive({ sortColumn: column, sortDirection: direction });
  }

  /** Limit the number of returned rows. Enables early termination. */
  limit(n: number): DataFrame<T> {
    if (n < 0) throw new Error("limit() must be non-negative");
    return this.derive({ limit: n });
  }

  /** Skip the first N rows. Enables offset-based pagination. */
  offset(n: number): DataFrame<T> {
    return this.derive({ offset: n });
  }

  /** Keyset pagination: fetch rows after the given cursor value. Requires sort(). */
  after(value: FilterOp["value"]): DataFrame<T> {
    if (!this._sortColumn) throw new Error("after() requires sort()");
    const op = this._sortDirection === "desc" ? "lt" : "gt";
    return this.derive({ filters: [...this._filters, { column: this._sortColumn, op, value }] });
  }

  /** Aggregate a column: .aggregate("sum", "amount") or .aggregate("count", "*") */
  aggregate(fn: AggregateOp["fn"], column: string, alias?: string): DataFrame {
    return this.derive({ aggregates: [...this._aggregates, { fn, column, alias }] }) as DataFrame;
  }

  /** Percentile aggregate: .percentile("amount", 0.95, "p95") */
  percentile(column: string, p: number, alias?: string): DataFrame {
    return this.derive({
      aggregates: [...this._aggregates, { fn: "percentile", column, alias, percentileTarget: p }],
    }) as DataFrame;
  }

  /** Group results by one or more columns. Used with .aggregate(). */
  groupBy(...columns: string[]): DataFrame<T> {
    return this.derive({ groupBy: columns });
  }

  /**
   * SIMD vector similarity search on a fixed_size_list column.
   * When queryVector is a string, the encoder is called client-side before query execution.
   */
  vector(
    column: string,
    queryVector: Float32Array | string,
    topK: number,
    opts?: VectorOpts,
  ): DataFrame<T> {
    if (typeof queryVector === "string") {
      if (!opts?.encoder) {
        throw new Error("vector() with a string query requires opts.encoder to convert text to Float32Array");
      }
      // Store the encoder and text — encoding happens at collect() time
      return this.derive({
        vectorQueryText: queryVector,
        vectorEncoder: opts.encoder,
        vectorSearch: {
          column,
          queryVector: new Float32Array(0), // resolved at collect() via encoder
          topK,
          metric: opts?.metric,
          nprobe: opts?.nprobe,
          efSearch: opts?.efSearch,
        },
      });
    }
    return this.derive({
      vectorSearch: {
        column,
        queryVector,
        topK,
        metric: opts?.metric,
        nprobe: opts?.nprobe,
        efSearch: opts?.efSearch,
      },
    });
  }

  /**
   * Join with another table query. Build side (right) is hashed, probe side (left) streams through.
   *
   * Usage:
   *   const orders = qm.table("orders");
   *   const users = qm.table("users");
   *   const result = await orders.join(users, { left: "user_id", right: "id" }).collect();
   */
  join(other: DataFrame, keys: JoinKeys, type?: JoinType): DataFrame {
    return this.derive({
      join: {
        right: other.toDescriptor(),
        leftKey: keys.left,
        rightKey: keys.right,
        type: type ?? "inner",
      },
    }) as DataFrame;
  }

  /** Enable query result caching with a TTL in milliseconds. */
  cache(opts: { ttl: number }): DataFrame<T> {
    return this.derive({ cacheTTL: opts.ttl });
  }

  /** Add a computed column — a plain callback that runs in-process at .collect() time. */
  computed(alias: string, fn: (row: T) => unknown): DataFrame {
    return this.derive({
      computedColumns: [...this._computedColumns, { alias, fn: fn as (row: Row) => unknown }],
    }) as DataFrame;
  }

  /** Add a window function. Serializable spec — no callbacks. */
  window(spec: WindowSpec): DataFrame {
    return this.derive({ windows: [...this._windows, spec] }) as DataFrame;
  }

  /** Deduplicate rows. If no columns specified, dedup on all columns. */
  distinct(...columns: string[]): DataFrame<T> {
    return this.derive({ distinct: columns.length > 0 ? columns : [] });
  }

  /** Union with another DataFrame. all=true for UNION ALL (no dedup). */
  union(other: DataFrame, all = false): DataFrame {
    return this.derive({
      setOperation: { mode: all ? "union_all" : "union", right: other.toDescriptor() },
    }) as DataFrame;
  }

  /** Intersect with another DataFrame. */
  intersect(other: DataFrame): DataFrame {
    return this.derive({
      setOperation: { mode: "intersect", right: other.toDescriptor() },
    }) as DataFrame;
  }

  /** Except (set difference) with another DataFrame. */
  except(other: DataFrame): DataFrame {
    return this.derive({
      setOperation: { mode: "except", right: other.toDescriptor() },
    }) as DataFrame;
  }

  /** IN subquery: filter where column values appear in another query's results. */
  filterIn(column: string, subquery: DataFrame): DataFrame<T> {
    return this.derive({
      deferredSubqueries: [...this._deferredSubqueries, {
        column,
        subquery: subquery.toDescriptor(),
      }],
    });
  }

  /** Query a specific version of a Lance dataset. */
  version(v: number): DataFrame<T> {
    return this.derive({ version: v });
  }

  // --- Execution (triggers computation) ---

  /**
   * Resolve deferred subqueries by executing them and building value sets.
   * Also resolves vector text queries via encoder.
   */
  private async resolveDeferred(): Promise<QueryDescriptor> {
    const desc = this.toDescriptor();

    // Resolve deferred subqueries into SubqueryInDefs with populated value sets
    if (this._deferredSubqueries.length > 0) {
      const resolved: { column: string; valueSet: Set<string> }[] = desc.subqueryIn ? [...desc.subqueryIn] : [];
      for (const deferred of this._deferredSubqueries) {
        const subResult = await this._executor.execute(deferred.subquery);
        const valueSet = new Set<string>();
        for (const row of subResult.rows) {
          // Use the first projected column, or the filter column from the subquery
          const cols = deferred.subquery.projections.length > 0
            ? deferred.subquery.projections
            : Object.keys(row);
          const col = cols[0];
          if (col) {
            const val = row[col];
            valueSet.add(val === null ? "__null__" : typeof val === "bigint" ? val.toString() : String(val));
          }
        }
        resolved.push({ column: deferred.column, valueSet });
      }
      desc.subqueryIn = resolved;
    }

    // Resolve vector text query via encoder
    if (this._vectorEncoder && this._vectorQueryText && desc.vectorSearch) {
      const encoded = await this._vectorEncoder(this._vectorQueryText);
      desc.vectorSearch = { ...desc.vectorSearch, queryVector: encoded };
    }

    return desc;
  }

  /** Execute and fully materialize results. */
  async collect(): Promise<QueryResult<T>> {
    const desc = await this.resolveDeferred();
    return this._executor.execute(desc) as Promise<QueryResult<T>>;
  }

  /** Alias for .collect() — backward compatibility with TableQuery.exec(). */
  async exec(): Promise<QueryResult<T>> {
    return this.collect();
  }

  /** Deferred page-at-a-time execution. Returns a LazyResult handle. */
  async lazy(): Promise<LazyResultHandle<T>> {
    const desc = await this.resolveDeferred();
    return new LazyResultHandle<T>(desc, this._executor);
  }

  /** Return the first matching row, or null. */
  async first(): Promise<T | null> {
    const desc = await this.resolveDeferred();
    if (this._executor.first && this._deferredSubqueries.length === 0 && !this._vectorEncoder) {
      return this._executor.first(desc) as Promise<T | null>;
    }
    desc.limit = 1;
    const result = await this._executor.execute(desc);
    return (result.rows[0] as T) ?? null;
  }

  /** Return the count of matching rows without full materialization. */
  async count(): Promise<number> {
    const desc = await this.resolveDeferred();
    if (this._executor.count && this._deferredSubqueries.length === 0 && !this._vectorEncoder) {
      return this._executor.count(desc);
    }
    desc.aggregates = [{ fn: "count", column: "*" }];
    const result = await this._executor.execute(desc);
    return (result.rows[0]?.["count_*"] as number) ?? 0;
  }

  /** Return true if at least one row matches. */
  async exists(): Promise<boolean> {
    const desc = await this.resolveDeferred();
    if (this._executor.exists && this._deferredSubqueries.length === 0 && !this._vectorEncoder) {
      return this._executor.exists(desc);
    }
    desc.limit = 1;
    const result = await this._executor.execute(desc);
    return result.rowCount > 0;
  }

  /** Inspect the query plan without executing. No data I/O is performed. */
  async explain(): Promise<ExplainResult> {
    if (!this._executor.explain) {
      throw new Error("explain() requires an executor with plan inspection support");
    }
    return this._executor.explain(this.toDescriptor());
  }

  /** Streaming iteration over results in batches. */
  async *stream(batchSize = 1000): AsyncGenerator<T[]> {
    if (this._executor.cursor && this._deferredSubqueries.length === 0 && !this._vectorEncoder) {
      for await (const batch of this._executor.cursor(this.toDescriptor(), batchSize)) {
        yield batch as T[];
      }
      return;
    }
    const result = await this.collect();
    for (let i = 0; i < result.rows.length; i += batchSize) {
      yield result.rows.slice(i, i + batchSize);
    }
  }

  /** Execute and cache results as a new DataFrame backed by the materialized result. */
  async materialize(): Promise<DataFrame<T>> {
    const result = await this.collect();
    return new DataFrame<T>(this._table, new MaterializedExecutor(result));
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

  /** Execute and return a columnar binary stream of rows. Only works with RemoteExecutor. */
  async execStream(): Promise<ReadableStream<Row>> {
    if (!this._executor.executeStream) {
      throw new Error("execStream() requires a remote executor with streaming support");
    }
    return this._executor.executeStream(this.toDescriptor());
  }

  toDescriptor(): QueryDescriptor {
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
      join: this._join,
      computedColumns: this._computedColumns.length > 0 ? this._computedColumns : undefined,
      windows: this._windows.length > 0 ? this._windows : undefined,
      distinct: this._distinct,
      setOperation: this._setOperation,
      subqueryIn: this._subqueryIn.length > 0
        ? this._subqueryIn.map(sq => ({ column: sq.column, valueSet: sq.valueSet }))
        : undefined,
      version: this._version,
    };
  }
}

/** Backward-compatible alias */
export type TableQuery<T extends Row = Row> = DataFrame<T>;
export const TableQuery = DataFrame;

// ---------------------------------------------------------------------------
// LazyResultHandle — deferred page-at-a-time execution
// ---------------------------------------------------------------------------

export class LazyResultHandle<T extends Row = Row> {
  private desc: QueryDescriptor;
  private executor: QueryExecutor;

  constructor(desc: QueryDescriptor, executor: QueryExecutor) {
    this.desc = desc;
    this.executor = executor;
  }

  /** Fetch a page of rows on demand. */
  async page(offset: number, limit: number): Promise<T[]> {
    const pageDesc = { ...this.desc, offset, limit };
    const result = await this.executor.execute(pageDesc);
    return result.rows as T[];
  }

  /** Fetch a single row. */
  async row(index: number): Promise<T> {
    const rows = await this.page(index, 1);
    return rows[0];
  }

  /** Full materialization. */
  async collect(): Promise<QueryResult<T>> {
    return this.executor.execute(this.desc) as Promise<QueryResult<T>>;
  }

  /** Streaming iteration. */
  async *stream(batchSize = 1000): AsyncGenerator<T[]> {
    let offset = 0;
    while (true) {
      const rows = await this.page(offset, batchSize);
      if (rows.length === 0) break;
      yield rows;
      offset += rows.length;
      if (rows.length < batchSize) break;
    }
  }
}

// ---------------------------------------------------------------------------
// MaterializedExecutor — executes queries against a pre-materialized result
// ---------------------------------------------------------------------------

class MaterializedExecutor implements QueryExecutor {
  private result: QueryResult;

  constructor(result: QueryResult) {
    this.result = result;
  }

  async execute(query: QueryDescriptor): Promise<QueryResult> {
    let rows = [...this.result.rows];

    // Apply filters
    for (const f of query.filters) {
      rows = rows.filter(row => {
        const v = row[f.column];
        if (v === null) return false;
        const fv = f.value;
        switch (f.op) {
          case "eq": return v === fv;
          case "neq": return v !== fv;
          case "gt": return typeof v === "number" && typeof fv === "number" ? v > fv : String(v) > String(fv);
          case "gte": return typeof v === "number" && typeof fv === "number" ? v >= fv : String(v) >= String(fv);
          case "lt": return typeof v === "number" && typeof fv === "number" ? v < fv : String(v) < String(fv);
          case "lte": return typeof v === "number" && typeof fv === "number" ? v <= fv : String(v) <= String(fv);
          case "in": return Array.isArray(fv) && (fv as unknown[]).includes(v);
          default: return true;
        }
      });
    }

    // Apply projections
    if (query.projections.length > 0) {
      const keep = new Set(query.projections);
      rows = rows.map(row => {
        const out: Row = {};
        for (const k of keep) if (k in row) out[k] = row[k];
        return out;
      });
    }

    // Apply offset + limit
    if (query.offset) rows = rows.slice(query.offset);
    if (query.limit !== undefined) rows = rows.slice(0, query.limit);

    return {
      rows,
      rowCount: rows.length,
      columns: query.projections.length > 0 ? query.projections : this.result.columns,
      bytesRead: 0,
      pagesSkipped: 0,
      durationMs: 0,
      cacheHit: true,
    };
  }
}

// ---------------------------------------------------------------------------
// DataFrameInit — internal state for derive()
// ---------------------------------------------------------------------------

interface DataFrameInit {
  filters: FilterOp[];
  projections: string[];
  sortColumn?: string;
  sortDirection?: "asc" | "desc";
  limit?: number;
  offset?: number;
  vectorSearch?: VectorSearchParams;
  aggregates: AggregateOp[];
  groupBy: string[];
  cacheTTL?: number;
  join?: JoinDescriptor;
  computedColumns: ComputedColumnDef[];
  windows: WindowSpec[];
  distinct?: string[];
  setOperation?: { mode: "union" | "union_all" | "intersect" | "except"; right: QueryDescriptor };
  subqueryIn: SubqueryInDef[];
  deferredSubqueries: DeferredSubquery[];
  version?: number;
  vectorEncoder?: (text: string) => Promise<Float32Array>;
  vectorQueryText?: string;
}

// ---------------------------------------------------------------------------
// QueryDescriptor — serializable query plan passed to executors
// ---------------------------------------------------------------------------

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
  join?: JoinDescriptor;
  // New fields:
  computedColumns?: { alias: string; fn: (row: Row) => unknown }[];
  windows?: WindowSpec[];
  distinct?: string[];
  setOperation?: { mode: "union" | "union_all" | "intersect" | "except"; right: QueryDescriptor };
  subqueryIn?: { column: string; valueSet: Set<string> }[];
  version?: number;
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
