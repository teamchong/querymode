# Spec: Public API

Status: **Implemented** — documents the current public contract.

All public surface area in one place. If it's not here, it's internal.

---

## 1. QueryMode (entry point)

```typescript
class QueryMode {
  // Construction
  static remote(queryDoNamespace, options?: { region?, locationHint?, masterDoNamespace? }): QueryMode
  static local(opts?: { wasmModule?, memoryBudgetBytes? }): QueryMode

  // Query
  table(name: string): DataFrame          // Start a query against a table/file
  sql(query: string): DataFrame           // Parse SQL → DataFrame
  query<T>(fn: () => Promise<T>): T       // Multi-table orchestration

  // Convenience constructors
  static fromJSON(data: object[], tableName?): DataFrame
  static fromCSV(csv: string, tableName?): Promise<DataFrame>
  static demo(tableName?): DataFrame

  // Metadata
  tables(): Promise<TableInfo[]>           // Edge mode only — list all tables
}
```

### Modes

| Mode | Construction | Storage | Use case |
|------|-------------|---------|----------|
| **Local** | `QueryMode.local()` | Filesystem / URLs | Node/Bun scripts, testing |
| **Edge** | `QueryMode.remote(env.QUERY_DO)` | R2 via Durable Objects | Production on Cloudflare Workers |

---

## 2. DataFrame (query builder)

Immutable. Every method returns a new DataFrame. Nothing executes until a terminal is called.

### Chainable Methods

```typescript
class DataFrame<T extends Row = Row> {
  // Filtering
  filter(column, op, value): DataFrame      // AND predicate
  or(column, op, value): DataFrame          // OR group
  where(column, op, value): DataFrame       // Alias for filter

  // Projection
  select(...columns): DataFrame
  withColumn(alias, fn): DataFrame          // Computed column
  rename(mapping: Record<string, string>): DataFrame

  // Sorting & Limiting
  sort(column, direction?): DataFrame       // "asc" | "desc"
  limit(n): DataFrame
  offset(n): DataFrame

  // Aggregation
  aggregate(agg: AggregateOp[]): DataFrame
  groupBy(...columns): DataFrame

  // Joins
  join(right: DataFrame, keys: JoinKeys, type?): DataFrame
  whereIn(column, subquery: DataFrame): DataFrame

  // Window Functions
  window(spec: WindowSpec): DataFrame

  // Set Operations
  union(other: DataFrame): DataFrame
  unionAll(other: DataFrame): DataFrame
  intersect(other: DataFrame): DataFrame
  except(other: DataFrame): DataFrame

  // Vector Search
  nearestTo(column, vector, topK, opts?): DataFrame

  // Time Travel
  version(n: number): DataFrame

  // Pandas-like sugar
  shape(): Promise<[number, number]>
  dtypes(): Promise<Record<string, DataType>>
  fillNull(value): DataFrame
  cast(column, dtype): DataFrame
  sample(n: number): DataFrame
  valueCounts(column): DataFrame

  // Misc
  distinct(...columns): DataFrame
  pipe(stage: PipeStage): DataFrame
  toCode(): string                          // Decompile to fluent builder code
}
```

### Terminal Methods (trigger execution)

```typescript
class DataFrame<T extends Row = Row> {
  exec(): Promise<QueryResult<T>>            // Full result with metadata
  collect(opts?: CollectOptions): Promise<T[]>  // Just rows
  first(): Promise<T | null>                 // First row or null
  count(): Promise<number>
  exists(): Promise<boolean>
  show(n?): Promise<string>                  // Pretty-printed table
  toJSON(): Promise<string>
  toCSV(): Promise<string>
  describe(): Promise<ExplainResult>         // Query plan inspection
  explain(): Promise<ExplainResult>          // Alias for describe

  // Write operations (edge mode only)
  append(rows, options?): Promise<AppendResult>
  drop(): Promise<DropResult>

  // Streaming
  stream(): Promise<ReadableStream<T>>
}
```

---

## 3. Filter Operators

All 14 ops support page-level and fragment-level pushdown.

| Op | Example | Notes |
|----|---------|-------|
| `eq` | `filter("status", "eq", "active")` | Exact match |
| `neq` | `filter("status", "neq", "deleted")` | Not equal |
| `gt` | `filter("age", "gt", 18)` | Greater than |
| `gte` | `filter("age", "gte", 18)` | Greater than or equal |
| `lt` | `filter("price", "lt", 100)` | Less than |
| `lte` | `filter("price", "lte", 100)` | Less than or equal |
| `in` | `filter("status", "in", ["a", "b"])` | In set |
| `not_in` | `filter("status", "not_in", ["x"])` | Not in set |
| `between` | `filter("age", "between", [18, 65])` | Inclusive range |
| `not_between` | `filter("age", "not_between", [0, 17])` | Outside range |
| `like` | `filter("name", "like", "%john%")` | SQL LIKE pattern |
| `not_like` | `filter("name", "not_like", "test%")` | Not matching pattern |
| `is_null` | `filter("email", "is_null", true)` | Is null |
| `is_not_null` | `filter("email", "is_not_null", true)` | Is not null |

---

## 4. Aggregate Functions

| Function | Column | Output Type | Notes |
|----------|--------|-------------|-------|
| `count` | `*` or column | number | COUNT(*) counts all rows, COUNT(col) counts non-null |
| `count_distinct` | column | number | Distinct non-null values |
| `sum` | numeric | number/bigint | Returns null for empty groups |
| `avg` | numeric | number | Returns null for empty groups |
| `min` | any | same as input | Returns null for empty groups |
| `max` | any | same as input | Returns null for empty groups |
| `stddev` | numeric | number | Sample standard deviation (Welford's algorithm) |
| `variance` | numeric | number | Sample variance |
| `median` | numeric | number | Exact median (sorts all values) |
| `percentile` | numeric | number | Requires `percentileTarget: 0..1` |

---

## 5. SQL Interface

### Entry Points

```typescript
QueryMode.sql("SELECT * FROM t WHERE x > 5")  // Returns DataFrame
sqlToDescriptor("SELECT ...")                   // Returns QueryDescriptor
parseSql("SELECT ...")                          // Returns AST
```

### Supported Syntax

```
SELECT [DISTINCT] expr [AS alias], ...
FROM table [alias]
[JOIN table [alias] ON condition]
[WHERE condition]
[GROUP BY expr, ...]
[HAVING condition]
[WINDOW ...]
[UNION [ALL] | INTERSECT | EXCEPT  SELECT ...]
[ORDER BY expr [ASC|DESC], ...]
[LIMIT n [OFFSET m]]
```

### Special Syntax

| Feature | Syntax | Notes |
|---------|--------|-------|
| Vector search | `WHERE NEAR(column, [0.1, ...], topK, 'metric')` | LanceDB-compatible |
| String concat | `col1 \|\| col2` | Not `+` |
| CASE | `CASE WHEN cond THEN val ELSE val END` | |
| CAST | `CAST(col AS TYPE)` | INT, BIGINT, FLOAT, DOUBLE, TEXT, BOOL |
| IN subquery | `WHERE col IN (SELECT ...)` | Materializes subquery first |
| WITH (CTE) | `WITH name AS (SELECT ...) SELECT ...` | Inlined, not materialized |

---

## 6. HTTP API (Cloudflare Workers)

Defined in `worker.ts`. All responses are JSON.

| Method | Path | Body | Response |
|--------|------|------|----------|
| POST | `/query` | `QueryDescriptor` | `QueryResult` |
| POST | `/sql` | `{ sql: string }` | `QueryResult` |
| POST | `/append` | `{ table, rows, options? }` | `AppendResult` |
| POST | `/drop` | `{ table }` | `DropResult` |
| GET | `/tables` | — | `{ tables: TableInfo[] }` |
| GET | `/health` | — | `{ ok: true }` |
| POST | `/upload` | multipart file | `AppendResult` (DEV_MODE only) |

---

## 7. PG Wire Protocol

Connect with any PostgreSQL client on the configured port.

```bash
psql -h localhost -p 5433 -U querymode
```

- Simple and extended query protocols
- Auth: trust or MD5
- Every SQL query goes through `sql/` → DataFrame → pipeline
- No transactions, no cursors, no COPY

---

## 8. RPC (Durable Object)

Zero-serialization via structured clone. Used by `RemoteExecutor`.

### QueryDO RPC

| Method | Input | Output |
|--------|-------|--------|
| `queryRpc(descriptor)` | QueryDescriptor | QueryResult |
| `countRpc(descriptor)` | QueryDescriptor | number |
| `existsRpc(descriptor)` | QueryDescriptor | boolean |
| `firstRpc(descriptor)` | QueryDescriptor | Row \| null |
| `explainRpc(descriptor)` | QueryDescriptor | ExplainResult |
| `streamRpc(descriptor)` | QueryDescriptor | ReadableStream\<Uint8Array\> |
| `listTablesRpc()` | — | { tables } |
| `getMetaRpc(table)` | string | TableMeta \| null |
| `invalidateRpc(payload)` | broadcast payload | void |

### MasterDO RPC

| Method | Input | Output |
|--------|-------|--------|
| `appendRpc(table, rows, options?)` | table + rows | AppendResult |
| `dropRpc(table)` | table name | DropResult |
| `listTablesRpc()` | — | { tables: string[] } |

---

## 9. Return Types

### QueryResult

```typescript
{
  rows: Row[]                    // Materialized result rows
  columnarData?: ArrayBuffer     // QMCB binary (when streaming between DOs)
  rowCount: number
  columns: string[]              // Columns that were fetched
  bytesRead: number              // Total R2 bytes
  pagesSkipped: number           // Pages skipped by pushdown
  durationMs: number
  // Timing breakdown
  r2ReadMs?: number
  wasmExecMs?: number
  // Cache stats
  cacheHits?: number             // L1: WASM buffer pool
  cacheMisses?: number
  edgeCacheHits?: number         // L2: caches.default
  edgeCacheMisses?: number
  cacheHit?: boolean             // Result-level cache hit
  // Spill stats
  spillBytesWritten?: number
  spillBytesRead?: number
}
```

### Row

```typescript
type Row = Record<string, number | bigint | string | boolean | Float32Array | null>
```

### ExplainResult

```typescript
{
  table: string
  format: "lance" | "parquet" | "iceberg"
  totalRows: number
  columns: { name, dtype, pages, bytes }[]
  pagesTotal: number
  pagesSkipped: number
  pagesScanned: number
  estimatedBytes: number
  estimatedR2Reads: number
  fragments: number
  fragmentsSkipped?: number
  partitionCatalog?: { column, partitionValues }
  filters: { column, op, pushable }[]
  metaCached: boolean
  estimatedRows: number
  fanOut?: boolean
  fragmentsScanned?: number
  hierarchicalReduction?: boolean
  reducerTiers?: number
}
```
