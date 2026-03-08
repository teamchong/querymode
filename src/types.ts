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
  frame?: { type: "rows" | "range"; start: number | "unbounded"; end: number | "current" | "unbounded" };
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

/** RPC interface exposed by MasterDO for zero-serialization calls */
export interface MasterDORpc {
  appendRpc(table: string, rows: Record<string, unknown>[], options?: AppendOptions): Promise<AppendResult>;
  dropRpc(table: string): Promise<DropResult>;
  registerRpc(queryDoId: string, region: string): Promise<{ registered: boolean; tableVersions?: Record<string, unknown> }>;
  writeRpc(body: unknown): Promise<unknown>;
  refreshRpc(body: unknown): Promise<unknown>;
  listTablesRpc(): Promise<{ tables: string[] }>;
}
