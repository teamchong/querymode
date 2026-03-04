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
  op: "eq" | "neq" | "gt" | "gte" | "lt" | "lte" | "in";
  value: number | bigint | string | (number | bigint | string)[];
}

/** Aggregation specification */
export interface AggregateOp {
  fn: "sum" | "avg" | "min" | "max" | "count";
  column: string;
  /** Output alias (defaults to fn_column, e.g., "sum_amount") */
  alias?: string;
}

/** Vector search parameters */
export interface VectorSearchParams {
  column: string;
  queryVector: Float32Array;
  topK: number;
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
  /** Number of page cache hits */
  cacheHits?: number;
  /** Number of page cache misses */
  cacheMisses?: number;
  /** Whether the result was served from cache */
  cacheHit?: boolean;
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

/** Result of an append (write) operation */
export interface AppendResult {
  version: number;
  dataFilePath: string;
  retries: number;
  rowsWritten: number;
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
  filters: { column: string; op: string; pushable: boolean }[];
  metaCached: boolean;
  /** Estimated number of rows after filter pushdown */
  estimatedRows: number;
}

/** Environment bindings for Cloudflare Workers */
export interface Env {
  DATA_BUCKET: R2Bucket;
  MASTER_DO: DurableObjectNamespace;
  QUERY_DO: DurableObjectNamespace;
  FRAGMENT_DO: DurableObjectNamespace;
}
