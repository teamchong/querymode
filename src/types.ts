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

/** Page-level metadata for skip/pushdown decisions */
export interface PageInfo {
  byteOffset: bigint;
  byteLength: number;
  rowCount: number;
  nullCount: number;
  minValue?: number | bigint | string;
  maxValue?: number | bigint | string;
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
  footer: Footer;
  columns: ColumnMeta[];
  totalRows: number;
  fileSize: bigint;
  /** R2 object key */
  r2Key: string;
  /** Timestamp of last footer update from Master DO */
  updatedAt: number;
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
  /** Number of IVF partitions to probe (default: sqrt(numPartitions)) */
  nprobe?: number;
}

/** Footer invalidation message from Master DO to regional Query DOs */
export interface FooterInvalidation {
  type: "footer_invalidation";
  table: string;
  r2Key: string;
  /** New footer bytes — Query DO doesn't need to re-read from R2 */
  footerBytes: ArrayBuffer;
  /** File size in bytes — needed by Query DO for cache metadata */
  fileSize: bigint;
  /** Timestamp of the write that triggered this invalidation */
  timestamp: number;
}

/** Query result row — typed at runtime from footer schema */
export type Row = Record<string, number | bigint | string | boolean | Float32Array | null>;

/** Execution result returned by .exec() */
export interface QueryResult {
  rows: Row[];
  rowCount: number;
  /** Columns that were actually fetched */
  columns: string[];
  /** Total bytes read from R2 */
  bytesRead: number;
  /** Number of pages skipped by filter pushdown */
  pagesSkipped: number;
  /** Query execution time in milliseconds */
  durationMs: number;
}

/** Environment bindings for Cloudflare Workers */
export interface Env {
  DATA_BUCKET: R2Bucket;
  MASTER_DO: DurableObjectNamespace;
  QUERY_DO: DurableObjectNamespace;
  LANCEQL_WASM: WebAssembly.Module;
}
