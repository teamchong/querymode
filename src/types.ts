/** Sentinel value used in GROUP BY / DISTINCT keys to distinguish null from empty string. */
export const NULL_SENTINEL = "\x01NULL\x01";

/** Build a GROUP BY key string from row values. Shared by partial-agg, executor, MaterializedExecutor, operators. */
export function groupKey(row: Row, cols: string[]): string {
  return groupKeyFrom(cols.length, (g) => row[cols[g]]);
}

/** Build a GROUP BY key from a value getter. Shared by columnar and QMCB aggregation paths. */
export function groupKeyFrom(count: number, getValue: (g: number) => unknown): string {
  let key = "";
  for (let g = 0; g < count; g++) {
    if (g > 0) key += "\0";
    const v = getValue(g);
    key += v === null || v === undefined ? NULL_SENTINEL : String(v);
  }
  return key;
}

/** Safely convert a number to BigInt — NaN/Infinity become 0n instead of throwing RangeError. */
export function safeBigInt(v: number): bigint {
  return Number.isFinite(v) ? BigInt(Math.trunc(v)) : 0n;
}

/** Nulls-last row comparator for a single sort column. */
export function rowComparator(col: string, desc: boolean): (a: Row, b: Row) => number {
  const dir = desc ? -1 : 1;
  return (a: Row, b: Row): number => {
    const av = a[col], bv = b[col];
    if ((av === null || av === undefined) && (bv === null || bv === undefined)) return 0;
    if (av === null || av === undefined) return 1;
    if (bv === null || bv === undefined) return -1;
    return av < bv ? -dir : av > bv ? dir : 0;
  };
}

/** Lance file footer metadata cached in DO memory */
export interface Footer {
  /** Byte offset where column metadata starts */
  columnMetaStart: bigint;
  /** Byte offset where column metadata offsets start */
  columnMetaOffsetsStart: bigint;
  /** Byte offset where global buffer offsets start */
  globalBuffOffsetsStart: bigint;
  /** Number of global buffers */
  numGlobalBuffers: number;
  /** Number of columns in the file */
  numColumns: number;
  /** Major version (2) */
  majorVersion: number;
  /** Minor version (0 or 1) */
  minorVersion: number;
}

/** Column metadata extracted from protobuf after footer */
export interface ColumnMeta {
  name: string;
  dtype: DataType;
  pages: PageInfo[];
  nullCount: number;
  /** For fixed_size_list: number of elements per row (e.g., 128 for 128-dim embedding) */
  listDimension?: number;
}

/** Count total rows across all pages of a column set. */
export function countColumnRows(columns: ColumnMeta[]): number {
  return columns[0]?.pages.reduce((s, p) => s + p.rowCount, 0) ?? 0;
}

/** Parquet-specific encoding info attached to pages */
export interface PageEncoding {
  encoding?: string;      // "PLAIN" | "RLE_DICTIONARY" | ...
  compression?: string;   // "UNCOMPRESSED" | "SNAPPY" | ...
  dictionaryPageOffset?: bigint;
  dictionaryPageLength?: number;
}

/** Page-level metadata for skip/pushdown decisions */
export interface PageInfo {
  byteOffset: bigint;
  byteLength: number;
  rowCount: number;
  nullCount: number;
  minValue?: number | bigint | string;
  maxValue?: number | bigint | string;
  /** Present only for Parquet pages */
  encoding?: PageEncoding;
  /** For Lance v2 nullable pages: byte offset where data starts within the page buffer (after bitmap + alignment padding) */
  dataOffsetInPage?: number;
}

export type DataType =
  | "int8"
  | "int16"
  | "int32"
  | "int64"
  | "uint8"
  | "uint16"
  | "uint32"
  | "uint64"
  | "float16"
  | "float32"
  | "float64"
  | "utf8"
  | "binary"
  | "bool"
  | "fixed_size_list";

/** Cached table metadata — everything needed to plan a query without touching R2 */
export interface TableMeta {
  name: string;
  /** Lance footer — only present for Lance format */
  footer?: Footer;
  format?: "lance" | "parquet" | "iceberg";
  columns: ColumnMeta[];
  totalRows: number;
  fileSize: bigint;
  /** R2 object key */
  r2Key: string;
  /** Timestamp of last footer update from Master DO */
  updatedAt: number;
  /** Vector indexes available for this table */
  vectorIndexes?: VectorIndexInfo[];
}

/** A filter predicate that can be pushed down to page-level skipping */
export interface FilterOp {
  column: string;
  op: "eq" | "neq" | "gt" | "gte" | "lt" | "lte" | "in" | "not_in" | "between" | "not_between" | "like" | "not_like" | "is_null" | "is_not_null";
  value: number | bigint | string | (number | bigint | string)[];
}

/** Aggregation specification */
export interface AggregateOp {
  fn: "sum" | "avg" | "min" | "max" | "count" | "count_distinct" | "stddev" | "variance" | "median" | "percentile";
  column: string;
  /** Output alias (defaults to fn_column, e.g., "sum_amount") */
  alias?: string;
  /** For percentile: target percentile (0-1) */
  percentileTarget?: number;
}

/** Window function specification (serializable — no callbacks) */
export interface WindowSpec {
  fn: "row_number" | "rank" | "dense_rank" | "lag" | "lead" | "sum" | "avg" | "min" | "max" | "count";
  /** Target column for lag/lead/sum/avg/min/max/count. Falls back to orderBy[0].column if not set. */
  column?: string;
  args?: { offset?: number; default_?: unknown };
  partitionBy: string[];
  orderBy: { column: string; direction: "asc" | "desc" }[];
  alias: string;
  frame?: { type: "rows" | "range"; start: number | "unbounded" | "current"; end: number | "current" | "unbounded" };
}

/**
 * Compute the set of column names referenced by a query descriptor.
 * Includes projections, filters, filterGroups, sort, groupBy, aggregates,
 * distinct, windows, join, subqueryIn, and vectorSearch.
 * Used by all scan paths to determine which columns to fetch from storage.
 */
export function queryReferencedColumns(query: {
  projections: string[];
  filters: FilterOp[];
  filterGroups?: FilterOp[][];
  sortColumn?: string;
  groupBy?: string[];
  aggregates?: AggregateOp[];
  distinct?: string[];
  windows?: WindowSpec[];
  join?: { leftKey: string };
  subqueryIn?: { column: string }[];
  computedColumns?: { alias: string }[];
  vectorSearch?: { column: string };
}, allColumnNames: string[]): Set<string> {
  // When computedColumns are present, read all columns — function bodies are opaque
  // and may reference any column (we can't statically analyze JS closures).
  if (query.computedColumns && query.computedColumns.length > 0) return new Set(allColumnNames);
  const s = new Set(query.projections.length > 0 ? query.projections : allColumnNames);
  for (const f of query.filters) s.add(f.column);
  if (query.filterGroups) for (const g of query.filterGroups) for (const f of g) s.add(f.column);
  if (query.sortColumn) s.add(query.sortColumn);
  if (query.groupBy) for (const g of query.groupBy) s.add(g);
  if (query.aggregates) for (const a of query.aggregates) if (a.column !== "*") s.add(a.column);
  if (query.distinct) for (const d of query.distinct) s.add(d);
  if (query.windows) for (const w of query.windows) {
    if (w.column) s.add(w.column);
    for (const p of w.partitionBy) s.add(p);
    for (const o of w.orderBy) s.add(o.column);
  }
  if (query.join) s.add(query.join.leftKey);
  if (query.subqueryIn) for (const sq of query.subqueryIn) s.add(sq.column);
  if (query.vectorSearch) s.add(query.vectorSearch.column);
  return s;
}

/** Join key specification */
export interface JoinKeys {
  left: string;
  right: string;
}

/** Join specification for code-first JOINs */
export interface JoinDescriptor {
  /** Query descriptor for the right side of the join */
  right: import("./client.js").QueryDescriptor;
  /** Left join key column */
  leftKey: string;
  /** Right join key column */
  rightKey: string;
  /** Join type (default: "inner") */
  type?: "inner" | "left" | "right" | "full" | "cross";
}

/** Join type union */
export type JoinType = "inner" | "left" | "right" | "full" | "cross";

/** Vector search parameters */
export interface VectorSearchParams {
  column: string;
  queryVector: Float32Array;
  topK: number;
  /** Distance metric (default: "cosine") */
  metric?: "cosine" | "l2" | "dot";
  /** IVF-PQ tuning: number of probes */
  nprobe?: number;
  /** HNSW tuning: search beam width */
  efSearch?: number;
}

/** Vector search options (client-facing, allows encoder) */
export interface VectorOpts {
  metric?: "cosine" | "l2" | "dot";
  nprobe?: number;
  efSearch?: number;
  /** Client-side encoder for text-to-vector conversion */
  encoder?: (text: string) => Promise<Float32Array>;
}

/** Version info for time travel */
export interface VersionInfo {
  version: number;
  timestamp: number;
  rowCount: number;
  fragmentCount: number;
}

/** Diff result for time travel */
export interface DiffResult {
  added: number;
  removed: number;
  addedFragments: string[];
  removedFragments: string[];
}

/** Query result row — typed at runtime from footer schema */
export type Row = Record<string, number | bigint | string | boolean | Float32Array | null>;

/** Execution result returned by .exec() */
export interface QueryResult<T extends Row = Row> {
  rows: T[];
  /** QMCB columnar binary — when present, rows may be empty (decode at consumer). */
  columnarData?: ArrayBuffer;
  rowCount: number;
  /** Columns that were actually fetched */
  columns: string[];
  /** Total bytes read from R2 */
  bytesRead: number;
  /** Number of pages skipped by filter pushdown */
  pagesSkipped: number;
  /** Query execution time in milliseconds */
  durationMs: number;
  /** Trace ID for correlating logs (cf-ray or UUID) */
  requestId?: string;
  /** Time spent fetching pages from R2 */
  r2ReadMs?: number;
  /** Time spent in WASM compute */
  wasmExecMs?: number;
  /** Number of WASM buffer pool cache hits (L1 — per-DO, in-memory) */
  cacheHits?: number;
  /** Number of WASM buffer pool cache misses (L1) */
  cacheMisses?: number;
  /** Whether the result was served from result cache */
  cacheHit?: boolean;
  /** Number of caches.default hits (L2 — per-datacenter, shared across DOs) */
  edgeCacheHits?: number;
  /** Number of caches.default misses (L2) */
  edgeCacheMisses?: number;
  /** Total bytes written to spill storage (R2 or filesystem) during sort/join */
  spillBytesWritten?: number;
  /** Total bytes read back from spill storage during sort/join */
  spillBytesRead?: number;
}

/** Schema field extracted from Lance manifest */
export interface SchemaField {
  name: string;
  logicalType: string;
  id: number;
  parentId: number;
  nullable: boolean;
}

/** Parsed Lance manifest — describes all fragments in a dataset version */
export interface ManifestInfo {
  version: number;
  fragments: FragmentInfo[];
  totalRows: number;
  schema: SchemaField[];
}

/** A single fragment (data file) within a Lance dataset */
export interface FragmentInfo {
  id: number;
  filePath: string;
  physicalRows: number;
}

/** Cached dataset metadata — multi-fragment Lance directories */
export interface DatasetMeta {
  name: string;
  r2Prefix: string;
  manifest: ManifestInfo;
  fragmentMetas: Map<number, TableMeta>;
  totalRows: number;
  updatedAt: number;
  /** User-specified partition column (set during partitioned ingest). */
  partitionBy?: string;
}

/** Iceberg table schema */
export interface IcebergSchema {
  fields: { name: string; type: string; required: boolean }[];
}

/** Cached Iceberg dataset metadata — multiple Parquet data files under one logical table */
export interface IcebergDatasetMeta {
  name: string;
  r2Prefix: string;
  schema: IcebergSchema;
  snapshotId: string;
  parquetFiles: string[];
  fragmentMetas: Map<number, TableMeta>;
  totalRows: number;
  updatedAt: number;
}

/** Result of a schema evolution operation */
export interface SchemaEvolutionResult {
  table: string;
  operation: "add_column" | "drop_column";
  column: string;
  columnsAfter: string[];
}

/** Options for append (write) operations — catalog-friendly */
export interface AppendOptions {
  /** R2 path prefix for the output Lance dataset (e.g., "pipelines/job-123/output.lance/").
   *  Defaults to "{table}.lance/" if not specified. */
  path?: string;
  /** Metadata attached to this write — visible to catalog layers for lineage/lifecycle. */
  metadata?: Record<string, string>;
  /** Partition column — rows are grouped by this column's value and written to separate fragments.
   *  Enables partition catalog pruning at query time for O(1) fragment lookup. */
  partitionBy?: string;
}

/** Result of an append (write) operation */
export interface AppendResult {
  version: number;
  dataFilePath: string;
  retries: number;
  rowsWritten: number;
  /** Metadata attached to this write, if any. */
  metadata?: Record<string, string>;
}

/** Result of a drop (delete) operation */
export interface DropResult {
  table: string;
  fragmentsDeleted: number;
  bytesFreed: number;
}

/** Vector index metadata for IVF-PQ acceleration */
export interface VectorIndexInfo {
  column: string;
  indexPath: string;
  type: "flat" | "ivf_pq";
  config?: { nPartitions: number; nSubvectors: number; nProbe: number };
}

/** Query plan inspection result returned by .explain() */
export interface ExplainResult {
  table: string;
  format: "lance" | "parquet" | "iceberg";
  totalRows: number;
  columns: { name: string; dtype: DataType; pages: number; bytes: number }[];
  pagesTotal: number;
  pagesSkipped: number;
  pagesScanned: number;
  estimatedBytes: number;
  estimatedR2Reads: number;
  fragments: number;
  /** Fragments skipped by fragment-level min/max pruning */
  fragmentsSkipped?: number;
  /** Partition catalog info (present when catalog exists for this table) */
  partitionCatalog?: { column: string; partitionValues: number };
  filters: { column: string; op: string; pushable: boolean }[];
  metaCached: boolean;
  /** Estimated number of rows after filter pushdown */
  estimatedRows: number;
  /** Whether the query fans out to Fragment DOs for parallel scan */
  fanOut?: boolean;
  /** Fragments scanned after pruning */
  fragmentsScanned?: number;
  /** Whether hierarchical reduction (tree merge) is used */
  hierarchicalReduction?: boolean;
  /** Number of reducer tiers in the tree merge (0 = flat) */
  reducerTiers?: number;
}

/** Environment bindings for Cloudflare Workers */
export interface Env {
  DATA_BUCKET: R2Bucket;
  /** Additional R2 buckets for PB-scale sharding (DATA_BUCKET_1, DATA_BUCKET_2, ...).
   *  When present, data is distributed across buckets by partition key hash. */
  DATA_BUCKET_1?: R2Bucket;
  DATA_BUCKET_2?: R2Bucket;
  DATA_BUCKET_3?: R2Bucket;
  MASTER_DO: DurableObjectNamespace;
  QUERY_DO: DurableObjectNamespace;
  FRAGMENT_DO: DurableObjectNamespace;
  /** Set to truthy in wrangler.toml [vars] to enable /upload endpoint. */
  DEV_MODE?: string;
}

/** RPC interface exposed by QueryDO for zero-serialization calls */
export interface QueryDORpc {
  queryRpc(descriptor: unknown): Promise<QueryResult>;
  countRpc(descriptor: unknown): Promise<number>;
  existsRpc(descriptor: unknown): Promise<boolean>;
  firstRpc(descriptor: unknown): Promise<Row | null>;
  explainRpc(descriptor: unknown): Promise<ExplainResult>;
  streamRpc(descriptor: unknown): Promise<ReadableStream<Uint8Array>>;
  invalidateRpc(payload: unknown): Promise<void>;
  listTablesRpc(): Promise<{ tables: unknown[] }>;
  getMetaRpc(table: string): Promise<TableMeta | null>;
  diagnosticsRpc(): Promise<Record<string, unknown>>;
  registerIcebergRpc(body: unknown): Promise<unknown>;
}

/**
 * FNV-1a cache key for a query descriptor — deterministic hash over all query fields.
 * Shared by local-executor and query-do for result cache dedup.
 */
export function queryCacheKey(query: {
  table: string;
  version?: number;
  filters: FilterOp[];
  filterGroups?: FilterOp[][];
  projections: string[];
  sortColumn?: string;
  sortDirection?: string;
  limit?: number;
  offset?: number;
  aggregates?: AggregateOp[];
  groupBy?: string[];
  distinct?: string[];
  windows?: WindowSpec[];
  computedColumns?: { alias: string; fn?: ((...args: never[]) => unknown) }[];
  setOperation?: { mode: string; right: unknown };
  subqueryIn?: { column: string; valueSet: Set<string> | string[] }[];
  join?: { type?: string; leftKey: string; rightKey: string; right: unknown };
}): string {
  let h = 0x811c9dc5;
  const feed = (s: string) => { for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); } };
  feed(query.table); feed("\0");
  if (query.version !== undefined) { feed(`v${query.version}`); feed("\0"); }
  for (const f of [...query.filters].sort((a, b) => a.column.localeCompare(b.column) || a.op.localeCompare(b.op))) {
    feed(f.column); feed("\0"); feed(f.op); feed("\0"); feed(String(f.value)); feed("\0");
  }
  if (query.filterGroups) {
    for (const group of query.filterGroups) {
      feed("|");
      for (const f of [...group].sort((a, b) => a.column.localeCompare(b.column) || a.op.localeCompare(b.op))) {
        feed(f.column); feed("\0"); feed(f.op); feed("\0"); feed(String(f.value)); feed("\0");
      }
    }
  }
  for (const p of [...query.projections].sort()) { feed(p); feed("\0"); }
  if (query.sortColumn) { feed(query.sortColumn); feed("\0"); feed(query.sortDirection ?? "asc"); feed("\0"); }
  if (query.limit !== undefined) { feed(String(query.limit)); feed("\0"); }
  if (query.offset !== undefined) { feed(String(query.offset)); feed("\0"); }
  if (query.aggregates) for (const a of query.aggregates) { feed(a.fn); feed("\0"); feed(a.column); feed("\0"); if (a.alias) feed(a.alias); feed("\0"); }
  if (query.groupBy) for (const g of query.groupBy) { feed(g); feed("\0"); }
  if (query.distinct) for (const d of query.distinct) { feed(d); feed("\0"); }
  if (query.windows) for (const w of query.windows) {
    feed(w.fn); feed("\0"); feed(w.alias); feed("\0"); feed(w.column ?? NULL_SENTINEL); feed("\0");
    if (w.partitionBy) for (const p of w.partitionBy) { feed(p); feed("\0"); }
    if (w.orderBy) for (const o of w.orderBy) { feed(o.column); feed(o.direction); feed("\0"); }
    if (w.frame) { feed(w.frame.type); feed(String(w.frame.start)); feed(String(w.frame.end)); feed("\0"); }
    if (w.args?.offset !== undefined) { feed(String(w.args.offset)); feed("\0"); }
    if (w.args?.default_ !== undefined) { feed(String(w.args.default_)); feed("\0"); }
  }
  if (query.computedColumns) for (const cc of query.computedColumns) { feed(cc.alias); feed("\0"); if (cc.fn) { feed(cc.fn.toString()); feed("\0"); } }
  if (query.setOperation) { feed(query.setOperation.mode); feed("\0"); feed(queryCacheKey(query.setOperation.right as Parameters<typeof queryCacheKey>[0])); feed("\0"); }
  if (query.subqueryIn) for (const sq of query.subqueryIn) { feed(sq.column); feed("\0"); for (const v of sq.valueSet) { feed(v); feed("\0"); } }
  if (query.join) { feed(query.join.type ?? "inner"); feed("\0"); feed(query.join.leftKey); feed("\0"); feed(query.join.rightKey); feed("\0"); feed(queryCacheKey(query.join.right as Parameters<typeof queryCacheKey>[0])); feed("\0"); }
  return `qr:${query.table}:${(h >>> 0).toString(36)}`;
}

/** RPC interface exposed by MasterDO for zero-serialization calls */
export interface MasterDORpc {
  appendRpc(table: string, rows: Record<string, unknown>[], options?: AppendOptions): Promise<AppendResult>;
  dropRpc(table: string): Promise<DropResult>;
  registerRpc(queryDoId: string, region: string): Promise<{ registered: boolean; tableVersions?: Record<string, unknown> }>;
  writeRpc(body: unknown): Promise<unknown>;
  refreshRpc(body: unknown): Promise<unknown>;
  listTablesRpc(): Promise<{ tables: string[] }>;
}
