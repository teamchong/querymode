import { DurableObject } from "cloudflare:workers";
import type { ColumnMeta, DataType, Env, ExplainResult, FilterOp, Footer, Row, TableMeta, DatasetMeta, IcebergDatasetMeta, QueryResult } from "./types.js";
import type { QueryDescriptor } from "./client.js";
import { parseFooter, parseColumnMetaFromProtobuf } from "./footer.js";
import { parseManifest, logicalTypeToDataType } from "./manifest.js";
import { detectFormat, getParquetFooterLength, parseParquetFooter, parquetMetaToTableMeta } from "./parquet.js";
import { parseIcebergMetadata, extractParquetPathsFromManifest } from "./iceberg.js";
import { canSkipPage, canSkipFragment, matchesFilter } from "./decode.js";
import { decodeParquetColumnChunk } from "./parquet-decode.js";
import { instantiateWasm, type WasmEngine } from "./wasm-engine.js";
import { mergeQueryResults } from "./merge.js";
import { coalesceRanges, autoCoalesceGap, fetchBounded, withRetry, withTimeout } from "./coalesce.js";
import { R2SpillBackend, encodeColumnarRun } from "./r2-spill.js";
import {
  type Operator, type RowBatch,
  buildEdgePipeline, drainPipeline, estimateRowSize,
  FilterOperator, HashJoinOperator, ProjectOperator,
  canSkipPageMultiCol, DEFAULT_MEMORY_BUDGET,
} from "./operators.js";
import { computePartialAgg, finalizePartialAgg } from "./partial-agg.js";
import { VipCache } from "./vip-cache.js";
import { parseLanceV2Columns } from "./lance-v2.js";
import { parseAndValidateQuery } from "./query-schema.js";
import { resolveBucket } from "./bucket.js";
import { PartitionCatalog } from "./partition-catalog.js";
import wasmModule from "./wasm-module.js";

const FRAGMENT_POOL_MAX = 100; // Max Fragment DO slots per datacenter (idle slots cost nothing)
const R2_TIMEOUT_MS = 10_000;
const FRAGMENT_TIMEOUT_MS = 25_000;
const FOOTER_CACHE_MAX = 1000; // ~4KB per footer = ~4MB at capacity
const VIP_THRESHOLD = 3; // Accesses needed to become "VIP" (protected from eviction)
const DATASET_CACHE_MAX = 100; // Max cached datasets before eviction
const RESULT_CACHE_MAX = 200; // Max cached query results
const RESULT_VIP_THRESHOLD = 2; // Accesses needed for VIP result cache
const EDGE_MEMORY_BUDGET = 32 * 1024 * 1024; // 32MB — DO has 128MB total, need room for WASM + caches

/**
 * EdgeScanOperator — yields RowBatch per page from R2, using 3-tier cache.
 * Wraps the same I/O logic as scanPages() but as a pull-based Operator.
 * WASM handles per-page decoding; TypeScript operators handle cross-page orchestration.
 */
class EdgeScanOperator implements Operator {
  private bucket: R2Bucket;
  private wasmEngine: WasmEngine;
  private meta: TableMeta;
  private query: QueryDescriptor;
  private footerCache: VipCache<string, TableMeta>;
  private edgeCacheGet: (r2Key: string, offset: number, length: number) => Promise<ArrayBuffer | null>;
  private edgeCachePut: (r2Key: string, offset: number, length: number, data: ArrayBuffer) => void;

  // Pre-computed per-column page info (after skip)
  private columnPageInfos = new Map<string, TableMeta["columns"][0]["pages"]>();
  private columnData = new Map<string, ArrayBuffer[]>();
  private cols: ColumnMeta[] = [];
  private fetched = false;
  private pageCount = 0;
  private currentPage = 0;

  // Stats
  bytesRead = 0;
  pagesSkipped = 0;
  cacheHits = 0;
  cacheMisses = 0;
  edgeCacheHits = 0;
  edgeCacheMisses = 0;
  r2ReadMs = 0;
  wasmExecMs = 0;
  /** Filters are applied inside WASM executeQuery — skip downstream FilterOperator. */
  filtersApplied = true;

  constructor(
    bucket: R2Bucket,
    wasmEngine: WasmEngine,
    meta: TableMeta,
    query: QueryDescriptor,
    footerCache: VipCache<string, TableMeta>,
    edgeCacheGet: (r2Key: string, offset: number, length: number) => Promise<ArrayBuffer | null>,
    edgeCachePut: (r2Key: string, offset: number, length: number, data: ArrayBuffer) => void,
  ) {
    this.bucket = bucket;
    this.wasmEngine = wasmEngine;
    this.meta = meta;
    this.query = query;
    this.footerCache = footerCache;
    this.edgeCacheGet = edgeCacheGet;
    this.edgeCachePut = edgeCachePut;
  }

  /** Compute column/page metadata once. Does NOT fetch page data. */
  private initColumns(): void {
    if (this.fetched) return;
    this.fetched = true;

    const query = this.query;
    const meta = this.meta;

    let cols = query.projections.length > 0
      ? meta.columns.filter(c => query.projections.includes(c.name))
      : meta.columns;

    if (query.vectorSearch) {
      const vc = query.vectorSearch.column;
      if (!cols.some(c => c.name === vc)) {
        const ec = meta.columns.find(c => c.name === vc);
        if (ec) cols = [...cols, ec];
      }
    }
    this.cols = cols;

    for (const col of cols) {
      const keptPages: typeof col.pages = [];
      for (const page of col.pages) {
        if (!query.vectorSearch && canSkipPage(page, query.filters, col.name)) {
          this.pagesSkipped++;
          continue;
        }
        keptPages.push(page);
      }
      this.columnPageInfos.set(col.name, keptPages);
    }

    // Count total pages from first column
    const firstCol = cols[0];
    if (firstCol) {
      const keptPages = this.columnPageInfos.get(firstCol.name) ?? firstCol.pages;
      this.pageCount = keptPages.length;
    }
  }

  /**
   * Fetch page data for a single page index across all columns.
   * Uses 3-tier cache: L1 (WASM pool) → L2 (caches.default) → L3 (R2).
   * Returns a Map of column name → ArrayBuffer for this page.
   */
  private async fetchPageData(pageIdx: number): Promise<Map<string, ArrayBuffer>> {
    const meta = this.meta;
    const result = new Map<string, ArrayBuffer>();
    const tableIsVip = this.footerCache.accessCount(this.query.table) >= VIP_THRESHOLD;

    // Build ranges for this single page across all columns
    type R = { column: string; offset: number; length: number };
    const ranges: R[] = [];
    for (const col of this.cols) {
      const keptPages = this.columnPageInfos.get(col.name) ?? col.pages;
      const page = keptPages[pageIdx];
      if (!page) continue;
      ranges.push({ column: col.name, offset: Number(page.byteOffset), length: page.byteLength });
    }

    // L1: WASM buffer pool
    const l1Misses: R[] = [];
    for (const r of ranges) {
      const cacheKey = `${meta.r2Key}:${r.offset}:${r.length}`;
      const cached = this.wasmEngine.cacheGet(cacheKey);
      if (cached) {
        this.cacheHits++;
        result.set(r.column, cached);
      } else {
        this.cacheMisses++;
        l1Misses.push(r);
      }
    }

    // L2: caches.default (VIP tables only)
    const uncachedRanges: R[] = [];
    if (tableIsVip && l1Misses.length > 0) {
      const edgeChecks = await Promise.all(
        l1Misses.map(async (r) => {
          const data = await this.edgeCacheGet(meta.r2Key, r.offset, r.length);
          return { range: r, data };
        }),
      );
      for (const { range: r, data } of edgeChecks) {
        if (data) {
          this.edgeCacheHits++;
          result.set(r.column, data);
          this.wasmEngine.cacheSet(`${meta.r2Key}:${r.offset}:${r.length}`, data);
        } else {
          this.edgeCacheMisses++;
          uncachedRanges.push(r);
        }
      }
    } else {
      for (const r of l1Misses) uncachedRanges.push(r);
    }

    // L3: R2
    if (uncachedRanges.length > 0) {
      const r2Start = Date.now();
      const coalesced = coalesceRanges(uncachedRanges, autoCoalesceGap(uncachedRanges));
      const fetched = await fetchBounded(
        coalesced.map(c => () =>
          withRetry(() =>
            withTimeout(
              (async () => {
                const obj = await this.bucket.get(meta.r2Key, { range: { offset: c.offset, length: c.length } });
                return obj ? { ...c, data: await obj.arrayBuffer() } : null;
              })(),
              R2_TIMEOUT_MS,
            ),
          ),
        ),
        8,
      );
      for (const f of fetched) {
        if (!f) continue;
        this.bytesRead += f.data.byteLength;
        for (const sub of f.ranges) {
          const slice = f.data.slice(sub.offset - f.offset, sub.offset - f.offset + sub.length);
          result.set(sub.column, slice);
          this.wasmEngine.cacheSet(`${meta.r2Key}:${sub.offset}:${sub.length}`, slice);
          if (tableIsVip) {
            this.edgeCachePut(meta.r2Key, sub.offset, sub.length, slice);
          }
        }
      }
      this.r2ReadMs += Date.now() - r2Start;
    }

    return result;
  }

  /**
   * Fetch ALL pages upfront. Used for Lance path which requires all pages
   * registered at once in WASM. NOT used for Parquet (which fetches per-page).
   */
  private async fetchAllPages(): Promise<void> {
    for (let pi = 0; pi < this.pageCount; pi++) {
      const pageData = await this.fetchPageData(pi);
      for (const [colName, buf] of pageData) {
        const arr = this.columnData.get(colName) ?? [];
        arr.push(buf);
        this.columnData.set(colName, arr);
      }
    }
  }

  async next(): Promise<RowBatch | null> {
    this.initColumns();

    if (this.meta.format === "parquet") {
      return this.nextParquetPage();
    }

    // Lance path: must register all pages at once (WASM constraint)
    if (this.currentPage > 0) return null;
    this.currentPage = 1;

    await this.fetchAllPages();

    const wasmStart = Date.now();
    this.wasmEngine.exports.resetHeap();
    const fragTable = `__edge_${this.meta.r2Key}`;
    for (const col of this.cols) {
      const pages = this.columnData.get(col.name);
      if (!pages?.length) continue;
      const keptPageInfos = this.columnPageInfos.get(col.name) ?? col.pages;
      if (!this.wasmEngine.registerColumn(fragTable, col.name, col.dtype, pages, keptPageInfos, col.listDimension)) {
        throw new Error(`WASM OOM: failed to register column "${col.name}"`);
      }
    }

    const decodeQuery: QueryDescriptor = {
      ...this.query,
      table: fragTable,
      sortColumn: undefined,
      limit: undefined,
      offset: undefined,
      aggregates: undefined,
      join: undefined,
    };
    const rows = this.wasmEngine.executeQuery(decodeQuery);
    if (!rows) throw new Error(`WASM query execution failed`);
    this.wasmEngine.clearTable(fragTable);
    this.columnData.clear(); // Release page buffers immediately
    this.wasmExecMs += Date.now() - wasmStart;

    return rows;
  }

  /** Yield one Parquet page at a time — fetches and decodes one page per call. */
  private async nextParquetPage(): Promise<RowBatch | null> {
    if (this.currentPage >= this.pageCount) return null;
    const pi = this.currentPage++;

    const cols = this.cols;
    const meta = this.meta;

    // Fetch page data for this single page (bounded: one page across all columns)
    const pageBuffers = await this.fetchPageData(pi);

    const wasmStart = Date.now();
    const colNames = cols.map(c => c.name);
    const decodedColumns = new Map<string, (number | bigint | string | boolean | null)[]>();

    for (const col of cols) {
      let chunkBuf = pageBuffers.get(col.name);
      if (!chunkBuf) { decodedColumns.set(col.name, []); continue; }
      const keptPageInfos = this.columnPageInfos.get(col.name) ?? col.pages;
      const pageInfo = keptPageInfos[pi];
      const encoding = pageInfo?.encoding ?? { compression: "UNCOMPRESSED" as const };

      if (encoding.dictionaryPageOffset !== undefined && encoding.dictionaryPageLength) {
        const dictOffset = Number(encoding.dictionaryPageOffset);
        const pageOffset = Number(pageInfo.byteOffset);
        if (dictOffset < pageOffset) {
          const dictKey = `${meta.r2Key}:${dictOffset}:${encoding.dictionaryPageLength}`;
          let dictBuf = this.wasmEngine.cacheGet(dictKey);
          if (!dictBuf) {
            const dictObj = await this.bucket.get(meta.r2Key, {
              range: { offset: dictOffset, length: encoding.dictionaryPageLength },
            });
            if (dictObj) {
              dictBuf = await dictObj.arrayBuffer();
              this.wasmEngine.cacheSet(dictKey, dictBuf);
              this.bytesRead += dictBuf.byteLength;
            }
          }
          if (dictBuf) {
            const combined = new Uint8Array(dictBuf.byteLength + chunkBuf.byteLength);
            combined.set(new Uint8Array(dictBuf), 0);
            combined.set(new Uint8Array(chunkBuf), dictBuf.byteLength);
            chunkBuf = combined.buffer;
          }
        }
      }

      const decoded = decodeParquetColumnChunk(
        chunkBuf, encoding, col.dtype, pageInfo?.rowCount ?? 0, this.wasmEngine,
      );
      decodedColumns.set(col.name, decoded);
    }

    // Try WASM SQL path: register decoded columns → executeQuery (SIMD filter/sort)
    let numRows = 0;
    for (const v of decodedColumns.values()) if (v.length > numRows) numRows = v.length;
    if (numRows === 0) return this.next(); // skip empty pages

    this.wasmEngine.exports.resetHeap();
    const fragTable = `__edge_pq_${pi}`;
    let wasmRegistered = true;
    for (const col of cols) {
      const values = decodedColumns.get(col.name);
      if (!values?.length) continue;
      if (!this.wasmEngine.registerDecodedColumn(fragTable, col.name, col.dtype, values)) {
        wasmRegistered = false;
        break;
      }
    }

    let rows: Row[];
    if (wasmRegistered) {
      const pqQuery: QueryDescriptor = {
        ...this.query,
        table: fragTable,
        sortColumn: undefined, limit: undefined, offset: undefined,
        aggregates: undefined, join: undefined,
      };
      const wasmRows = this.wasmEngine.executeQuery(pqQuery);
      this.wasmEngine.clearTable(fragTable);
      rows = wasmRows ?? [];
    } else {
      // Fallback: assemble rows in JS
      rows = [];
      for (let i = 0; i < numRows; i++) {
        const row: Row = {};
        for (const name of colNames) {
          const vals = decodedColumns.get(name);
          row[name] = vals && i < vals.length ? vals[i] : null;
        }
        rows.push(row);
      }
    }

    this.wasmExecMs += Date.now() - wasmStart;
    return rows.length > 0 ? rows : this.next();
  }

  async close(): Promise<void> {
    this.columnData.clear();
  }
}

/**
 * Query DO — per-region reader with cached footers.
 * WASM (Zig SIMD) handles per-page compute; TS operators handle cross-page orchestration.
 */
export class QueryDO extends DurableObject<Env> {
  private footerCache = new VipCache<string, TableMeta>(FOOTER_CACHE_MAX, VIP_THRESHOLD);
  private datasetCache = new Map<string, DatasetMeta>();
  private partitionCatalogs = new Map<string, PartitionCatalog>();
  private resultCache = new VipCache<string, QueryResult>(RESULT_CACHE_MAX, RESULT_VIP_THRESHOLD);
  private wasmEngine!: WasmEngine;
  private activeFragmentSlots = new Set<number>(); // slots currently scanning
  private initialized = false;
  private registeredWithMaster = false;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
  }

  /** Shorthand for multi-bucket resolution. */
  private r2(r2Key: string): R2Bucket {
    return resolveBucket(this.env, r2Key);
  }

  private log(level: "info" | "warn" | "error", msg: string, data?: Record<string, unknown>): void {
    console[level === "error" ? "error" : level === "warn" ? "warn" : "log"](
      JSON.stringify({ ts: new Date().toISOString(), level, msg, ...data }),
    );
  }

  /** Assemble Row[] from a column-major decoded map. */
  private assembleRowsFromColumns(
    decodedColumns: Map<string, (number | bigint | string | boolean | null)[]>,
    colNames: string[],
  ): Row[] {
    let numRows = 0;
    for (const v of decodedColumns.values()) if (v.length > numRows) numRows = v.length;
    const rows: Row[] = [];
    for (let i = 0; i < numRows; i++) {
      const row: Row = {};
      for (const name of colNames) {
        const vals = decodedColumns.get(name);
        row[name] = vals && i < vals.length ? vals[i] : null;
      }
      rows.push(row);
    }
    return rows;
  }

  /** Apply filters, sort, and limit in JS. Handles bigint↔number coercion for cross-type comparison. */
  private applyJsPostProcessing(rows: Row[], query: QueryDescriptor): Row[] {
    let result = rows;
    if (query.filters.length > 0) {
      result = result.filter(row =>
        query.filters.every(f => {
          const v = row[f.column];
          if (f.op === "is_null") return v === null || v === undefined;
          if (f.op === "is_not_null") return v !== null && v !== undefined;
          if (v === null || v === undefined) return false;
          return matchesFilter(v as number | bigint | string | boolean, f);
        }),
      );
    }
    if (query.filterGroups && query.filterGroups.length > 0) {
      result = result.filter(row =>
        query.filterGroups!.some(group =>
          group.every(f => {
            const v = row[f.column];
            if (f.op === "is_null") return v === null || v === undefined;
            if (f.op === "is_not_null") return v !== null && v !== undefined;
            if (v === null || v === undefined) return false;
            return matchesFilter(v as number | bigint | string | boolean, f);
          }),
        ),
      );
    }
    if (query.sortColumn) {
      const dir = query.sortDirection === "desc" ? -1 : 1;
      const sc = query.sortColumn;
      result.sort((a, b) => {
        const va = a[sc], vb = b[sc];
        if (va == null && vb == null) return 0;
        if (va == null) return dir;
        if (vb == null) return -dir;
        return va < vb ? -dir : va > vb ? dir : 0;
      });
    }
    const offset = query.offset ?? 0;
    if (offset > 0 || (query.limit && query.limit > 0)) {
      result = result.slice(offset, query.limit ? offset + query.limit : undefined);
    }
    return result;
  }

  /** Evict oldest dataset entry (by updatedAt) when cache exceeds max size. */
  /** Auto-detect best partition column and build catalog from fragment metadata. */
  private buildPartitionCatalog(tableName: string, fragmentMetas: Map<number, TableMeta>, partitionBy?: string): void {
    if (fragmentMetas.size < 2) return; // no point for single-fragment tables

    // User-specified partition column takes priority
    if (partitionBy) {
      const catalog = PartitionCatalog.fromFragments(partitionBy, fragmentMetas);
      this.partitionCatalogs.set(tableName, catalog);
      void this.ctx.storage.put(`pcatalog:${tableName}`, catalog.serialize());
      this.log("info", "partition_catalog_built", { ...catalog.stats(), source: "explicit" });
      return;
    }

    // Auto-detect: find column with the most distinct min/max values across fragments
    const firstMeta = fragmentMetas.values().next().value;
    if (!firstMeta) return;

    let bestColumn = "";
    let bestScore = 0;
    const fragCount = fragmentMetas.size;

    for (const col of firstMeta.columns) {
      if (col.pages.length === 0) continue;
      const values = new Set<string>();
      for (const [, meta] of fragmentMetas) {
        const c = meta.columns.find(mc => mc.name === col.name);
        if (!c) continue;
        for (const page of c.pages) {
          if (page.minValue !== undefined) values.add(String(page.minValue));
          if (page.maxValue !== undefined) values.add(String(page.maxValue));
        }
      }
      if (values.size < 2 || values.size >= 10_000) continue;
      // Good partition key: multiple fragments per value (not near-unique).
      // Score = distinct values, penalized if ratio of values-to-fragments is too high
      // (e.g., 9000 values across 10 fragments = useless, 50 values across 10 fragments = great)
      const ratio = values.size / fragCount;
      if (ratio > 100) continue; // too many unique values per fragment — skip (likely timestamps/IDs)
      const score = values.size / (1 + ratio); // reward cardinality, penalize high ratio
      if (score > bestScore) {
        bestScore = score;
        bestColumn = col.name;
      }
    }

    if (bestColumn) {
      const catalog = PartitionCatalog.fromFragments(bestColumn, fragmentMetas);
      this.partitionCatalogs.set(tableName, catalog);
      void this.ctx.storage.put(`pcatalog:${tableName}`, catalog.serialize());
      this.log("info", "partition_catalog_built", catalog.stats());
    }
  }

  private evictDatasetCache(): void {
    if (this.datasetCache.size <= DATASET_CACHE_MAX) return;
    let oldestKey: string | null = null;
    let oldestTime = Infinity;
    for (const [key, meta] of this.datasetCache) {
      if (meta.updatedAt < oldestTime) {
        oldestTime = meta.updatedAt;
        oldestKey = key;
      }
    }
    if (oldestKey) this.datasetCache.delete(oldestKey);
  }

  private async ensureInitialized(): Promise<void> {
    if (this.initialized) return;
    this.initialized = true;

    const stored = await this.ctx.storage.list<TableMeta>({ prefix: "table:" });
    for (const [key, meta] of stored) this.footerCache.set(key.replace("table:", ""), meta);

    // Restore persisted partition catalogs
    const catalogs = await this.ctx.storage.list<ReturnType<PartitionCatalog["serialize"]>>({ prefix: "pcatalog:" });
    for (const [key, data] of catalogs) {
      const tableName = key.replace("pcatalog:", "");
      this.partitionCatalogs.set(tableName, PartitionCatalog.deserialize(data));
    }

    this.wasmEngine = await instantiateWasm(wasmModule);

    // Register with Master for invalidation broadcasts
    this.registerWithMaster();
  }

  private registerWithMaster(): void {
    const master = this.env.MASTER_DO.get(this.env.MASTER_DO.idFromName("master")) as unknown as import("./types.js").MasterDORpc;
    this.ctx.storage.get<string>("region").then(async (region) => {
      try {
        const body = await master.registerRpc(this.ctx.id.toString(), region ?? "unknown");
        this.registeredWithMaster = true;
        // Master returns current table timestamps — refresh any stale entries
        // This closes the gap where broadcasts were missed during hibernation
        if (body.tableVersions) {
          await this.refreshStaleTables(body.tableVersions as Record<string, { r2Key: string; updatedAt: number }>);
        }
      } catch (err) {
        this.log("warn", "master_register_failed", {
          note: "will retry on next request",
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }).catch((err) => {
      this.log("warn", "master_register_storage_error", {
        error: err instanceof Error ? err.message : String(err),
      });
    });
  }

  /** Compare Master's table timestamps against local cache, refresh any that are stale. */
  private async refreshStaleTables(
    masterVersions: Record<string, { r2Key: string; updatedAt: number }>,
  ): Promise<void> {
    for (const [table, { r2Key, updatedAt }] of Object.entries(masterVersions)) {
      const cached = this.footerCache.get(table);
      if (cached && cached.updatedAt >= updatedAt) continue;

      // Stale or missing — re-read footer from R2
      const head = await this.r2(r2Key).head(r2Key);
      if (!head) continue;

      const fileSize = BigInt(head.size);
      const tailSize = Math.min(Number(fileSize), 40);
      const obj = await this.r2(r2Key).get(r2Key, { range: { offset: Number(fileSize) - tailSize, length: tailSize } });
      if (!obj) continue;

      const tailBuf = await obj.arrayBuffer();
      const fmt = detectFormat(tailBuf);

      let meta: TableMeta;
      if (fmt === "parquet") {
        const footerLen = getParquetFooterLength(tailBuf);
        if (!footerLen) continue;
        const footerObj = await this.r2(r2Key).get(r2Key, {
          range: { offset: Number(fileSize) - footerLen - 8, length: footerLen },
        });
        if (!footerObj) continue;
        const parquetMeta = parseParquetFooter(await footerObj.arrayBuffer());
        if (!parquetMeta) continue;
        meta = parquetMetaToTableMeta(parquetMeta, r2Key, fileSize);
        meta.name = table;
        meta.updatedAt = updatedAt;
      } else {
        const footer = parseFooter(tailBuf);
        if (!footer) continue;
        const columns = await this.readColumnMeta(r2Key, footer);
        meta = {
          name: table, footer, format: "lance", columns,
          totalRows: columns[0]?.pages.reduce((s, p) => s + p.rowCount, 0) ?? 0,
          fileSize, r2Key, updatedAt,
        };
      }

      this.footerCache.set(table, meta);
      this.wasmEngine.clearTable(table);
      await this.ctx.storage.put(`table:${table}`, meta);
    }
  }

  private async executeInvalidation(body: {
    table: string; r2Key: string; footerRaw: ArrayBuffer;
    columns?: ColumnMeta[]; fileSize: bigint; timestamp: number;
    format?: "lance" | "parquet" | "iceberg";
    totalRows?: number; r2Prefix?: string;
  }): Promise<void> {
    // If broadcast includes r2Prefix, this is a dataset — lazy-load full dataset from R2
    if (body.r2Prefix) {
      this.datasetCache.delete(body.table);
      this.resultCache.invalidateByPrefix(`qr:${body.table}:`);
      this.wasmEngine.clearTable(body.table);
      this.wasmEngine.cacheClear();
      // Trigger lazy-load of the full dataset (reads manifest + all fragments)
      const dataset = await this.loadDatasetFromR2(body.table);
      if (dataset) return;
      // Fall through to single-fragment cache if dataset load fails
    }

    const fmt = body.format ?? "lance";
    let columns: ColumnMeta[];
    let footer: Footer | undefined;

    if (fmt === "parquet" && body.columns) {
      // Parquet invalidation — columns already parsed by Master
      columns = body.columns;
    } else {
      // Lance invalidation — parse footer from raw ArrayBuffer (zero-copy via RPC)
      const parsed = parseFooter(body.footerRaw);
      if (!parsed) throw new Error("Invalid footer");
      footer = parsed;
      columns = body.columns ?? await this.readColumnMeta(body.r2Key, parsed);
    }

    // Use totalRows from broadcast if available and > 0, otherwise fall back to column computation
    const computedRows = columns[0]?.pages.reduce((s, p) => s + p.rowCount, 0) ?? 0;
    const totalRows = (body.totalRows != null && body.totalRows > 0) ? body.totalRows : computedRows;

    const meta: TableMeta = {
      name: body.table, footer, format: fmt, columns,
      totalRows,
      fileSize: body.fileSize,
      r2Key: body.r2Key, updatedAt: body.timestamp,
    };

    this.footerCache.set(body.table, meta);
    this.resultCache.invalidateByPrefix(`qr:${body.table}:`);
    this.wasmEngine.clearTable(body.table);
    this.wasmEngine.cacheClear();
    await this.ctx.storage.put(`table:${body.table}`, meta);
  }

  private queryKey(query: QueryDescriptor): string {
    // FNV-1a hash over query components — no JSON serialization
    let h = 0x811c9dc5;
    const feed = (s: string) => { for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); } };
    feed(query.table);
    for (const f of [...query.filters].sort((a, b) => a.column.localeCompare(b.column) || a.op.localeCompare(b.op))) {
      feed(f.column); feed(f.op); feed(String(f.value));
    }
    if (query.filterGroups) {
      for (const group of query.filterGroups) {
        feed("|"); // group separator
        for (const f of [...group].sort((a, b) => a.column.localeCompare(b.column) || a.op.localeCompare(b.op))) {
          feed(f.column); feed(f.op); feed(String(f.value));
        }
      }
    }
    for (const p of [...query.projections].sort()) feed(p);
    if (query.sortColumn) { feed(query.sortColumn); feed(query.sortDirection ?? "asc"); }
    if (query.limit !== undefined) feed(String(query.limit));
    if (query.aggregates) for (const a of query.aggregates) { feed(a.fn); feed(a.column); if (a.alias) feed(a.alias); }
    if (query.groupBy) for (const g of query.groupBy) feed(g);
    return `qr:${query.table}:${(h >>> 0).toString(36)}`;
  }

  private parseQuery(body: unknown): QueryDescriptor {
    return parseAndValidateQuery(body) as QueryDescriptor;
  }

  /** Core count logic shared by HTTP handler and RPC. */
  private async executeCount(query: QueryDescriptor): Promise<number> {
    if (query.filters.length === 0) {
      const meta = this.footerCache.get(query.table)
        ?? (await this.loadTableFromR2(query.table)) ?? undefined;
      if (meta) return meta.columns[0]?.pages.reduce((s, p) => s + p.rowCount, 0) ?? meta.totalRows;
    }
    query.aggregates = [{ fn: "count", column: "*" }];
    const result = await this.executeQuery(query);
    return (result.rows[0]?.["count_*"] as number) ?? 0;
  }


  /** Core explain logic. */
  private async executeExplain(query: QueryDescriptor): Promise<ExplainResult> {
    let meta: TableMeta | undefined = this.footerCache.get(query.table);
    const metaCached = !!meta;
    if (!meta) {
      meta = (await this.loadTableFromR2(query.table)) ?? undefined;
      if (!meta) throw new Error(`Table "${query.table}" not found`);
    }

    const { columns } = meta;
    const projectedColumns = query.projections.length > 0
      ? columns.filter(c => query.projections.includes(c.name))
      : columns;

    let pagesTotal = 0;
    let pagesSkipped = 0;
    const ranges: { column: string; offset: number; length: number }[] = [];
    const colDetails: ExplainResult["columns"] = [];

    for (const col of projectedColumns) {
      let colBytes = 0;
      let colPages = 0;
      for (const page of col.pages) {
        pagesTotal++;
        if (!query.vectorSearch && canSkipPage(page, query.filters, col.name)) {
          pagesSkipped++;
          continue;
        }
        colPages++;
        colBytes += page.byteLength;
        ranges.push({ column: col.name, offset: Number(page.byteOffset), length: page.byteLength });
      }
      colDetails.push({ name: col.name, dtype: col.dtype as DataType, pages: colPages, bytes: colBytes });
    }

    const coalesced = coalesceRanges(ranges, autoCoalesceGap(ranges));
    const estimatedBytes = ranges.reduce((s, r) => s + r.length, 0);
    const dataset = this.datasetCache.get(query.table);
    const totalFragments = dataset ? dataset.fragmentMetas.size : 1;
    let fragmentsSkipped = 0;
    if (dataset) {
      for (const fragMeta of dataset.fragmentMetas.values()) {
        if (canSkipFragment(fragMeta, query.filters, query.filterGroups)) {
          fragmentsSkipped++;
        }
      }
    }

    return {
      table: query.table,
      format: meta.format ?? "lance",
      totalRows: meta.totalRows,
      columns: colDetails,
      pagesTotal,
      pagesSkipped,
      pagesScanned: pagesTotal - pagesSkipped,
      estimatedBytes,
      estimatedR2Reads: coalesced.length,
      estimatedRows: meta.totalRows,
      fragments: totalFragments,
      fragmentsSkipped,
      partitionCatalog: this.partitionCatalogs.has(query.table)
        ? { column: this.partitionCatalogs.get(query.table)!.column, partitionValues: this.partitionCatalogs.get(query.table)!.stats().partitionValues }
        : undefined,
      filters: [
        ...query.filters.map(f => ({
          column: f.column,
          op: f.op,
          pushable: f.op !== "in" && f.op !== "neq",
        })),
        ...(query.filterGroups ?? []).flatMap(group =>
          group.map(f => ({
            column: f.column,
            op: f.op,
            pushable: f.op !== "in" && f.op !== "neq",
          })),
        ),
      ],
      metaCached,
    };
  }

  /** Execute query using page-level R2 Range reads + WASM compute. Never downloads full files. */
  private async executeQuery(query: QueryDescriptor): Promise<QueryResult> {
    const t0 = Date.now();

    // Result cache check (skip for vector search)
    const cacheable = !!(query.cacheTTL && !query.vectorSearch);
    if (cacheable) {
      const cached = this.resultCache.get(this.queryKey(query));
      if (cached) return { ...cached, durationMs: 0 };
    }

    // Join path: use operator pipeline with R2 spill
    if (query.join) {
      const result = await this.executeJoin(query, t0);
      if (cacheable) {
        this.resultCache.setWithTTL(this.queryKey(query), result, query.cacheTTL!);
      }
      return result;
    }

    // Multi-fragment dataset path
    const dataset = this.datasetCache.get(query.table);
    let result: QueryResult;
    if (dataset) {
      result = await this.executeMultiFragment(query, dataset, t0);
    } else {
      let meta: TableMeta | undefined = this.footerCache.get(query.table);
      if (!meta) {
        meta = (await this.loadTableFromR2(query.table)) ?? undefined;
        if (!meta) throw new Error(`Table "${query.table}" not found`);
      }

      // Use operator pipeline for any query that could produce unbounded results:
      // - ORDER BY without LIMIT (needs external sort with spill)
      // - No LIMIT at all (unbounded scan — needs streaming)
      // - Has aggregates without LIMIT (high-cardinality GROUP BY could be large)
      // Exceptions: vector search (uses IVF-PQ index, always bounded by topK)
      const hasLimit = query.limit !== undefined;
      const hasAgg = query.aggregates && query.aggregates.length > 0;
      // Aggregation must scan all matching rows before applying LIMIT to the output.
      // Route through the pipeline so LIMIT applies after aggregation, not before.
      const needsPipeline = !query.vectorSearch && (!hasLimit || hasAgg);
      if (needsPipeline) {
        result = await this.executeWithPipeline(query, meta, t0);
      } else {
        // Bounded queries (has LIMIT or vector search) — safe to materialize in WASM
        const hasPages = meta.columns.some(c => c.pages.length > 0);
        if (!hasPages && meta.format === "lance" && meta.r2Key) {
          result = await this.executeLanceWholeFile(query, meta, t0);
        } else {
          result = await this.scanPages(query, meta, t0);
        }
      }
    }

    if (cacheable) {
      this.resultCache.setWithTTL(this.queryKey(query), result, query.cacheTTL!);
    }
    return result;
  }

  /** Load entire Lance fragment into WASM and execute SQL. Used when page-level metadata is unavailable. */
  private async executeLanceWholeFile(query: QueryDescriptor, meta: TableMeta, t0: number): Promise<QueryResult> {
    const r2Start = Date.now();
    let bytesRead = 0;

    // Check WASM buffer pool cache first
    const cacheKey = `lance:${meta.r2Key}`;
    let fileData = this.wasmEngine.cacheGet(cacheKey);
    let cacheHit = !!fileData;

    if (!fileData) {
      const obj = await this.r2(meta.r2Key).get(meta.r2Key);
      if (!obj) throw new Error(`Failed to read Lance file: ${meta.r2Key}`);
      fileData = await obj.arrayBuffer();
      bytesRead = fileData.byteLength;
      this.wasmEngine.cacheSet(cacheKey, fileData);
    }
    const r2ReadMs = Date.now() - r2Start;

    const wasmStart = Date.now();
    this.wasmEngine.exports.resetHeap();

    // Load fragment and extract columns via fragment reader
    const dataPtr = this.wasmEngine.exports.alloc(fileData.byteLength);
    if (!dataPtr) throw new Error("WASM OOM allocating Lance file buffer");
    new Uint8Array(this.wasmEngine.exports.memory.buffer, dataPtr, fileData.byteLength)
      .set(new Uint8Array(fileData));

    const loadResult = this.wasmEngine.exports.fragmentLoad(dataPtr, fileData.byteLength);
    if (loadResult === 0) throw new Error(`Failed to load Lance fragment (invalid file?)`);

    // Parse Lance v2 column metadata using shared parser
    const dataset = this.datasetCache.get(query.table);
    const schema = dataset?.manifest.schema;
    const colInfos = parseLanceV2Columns(fileData, schema, meta.totalRows);
    if (!colInfos || colInfos.length === 0) {
      throw new Error("Failed to parse Lance v2 column metadata");
    }

    this.log("info", "lance_fragment_parsed", {
      numCols: colInfos.length,
      colInfos: colInfos.map(c => ({ name: c.name, dtype: c.dtype, rows: c.rowCount })),
    });

    // Read raw column data from the Lance file and assemble rows
    // For simple Lance files, data is at the beginning of the file in column-major order
    let dataPos = 0;
    const decodedColumns = new Map<string, (number | bigint | string | boolean | null)[]>();
    for (const col of colInfos) {
      if (query.projections.length > 0 && !query.projections.includes(col.name)) {
        // Skip this column's data
        if (col.dtype === "int64" || col.dtype === "float64") dataPos += col.rowCount * 8;
        else if (col.dtype === "int32" || col.dtype === "float32") dataPos += col.rowCount * 4;
        continue;
      }

      const values: (number | bigint | string | boolean | null)[] = [];
      const dv = new DataView(fileData, dataPos);
      if (col.dtype === "int64") {
        for (let i = 0; i < col.rowCount; i++) values.push(dv.getBigInt64(i * 8, true));
        dataPos += col.rowCount * 8;
      } else if (col.dtype === "float64") {
        for (let i = 0; i < col.rowCount; i++) values.push(dv.getFloat64(i * 8, true));
        dataPos += col.rowCount * 8;
      } else if (col.dtype === "int32") {
        for (let i = 0; i < col.rowCount; i++) values.push(dv.getInt32(i * 4, true));
        dataPos += col.rowCount * 4;
      } else if (col.dtype === "float32") {
        for (let i = 0; i < col.rowCount; i++) values.push(dv.getFloat32(i * 4, true));
        dataPos += col.rowCount * 4;
      }
      decodedColumns.set(col.name, values);
    }

    const colNames = [...decodedColumns.keys()];
    let rows = this.applyJsPostProcessing(
      this.assembleRowsFromColumns(decodedColumns, colNames), query,
    );
    const wasmExecMs = Date.now() - wasmStart;

    return {
      rows, rowCount: rows.length, columns: meta.columns.map(c => c.name),
      bytesRead, pagesSkipped: 0, durationMs: Date.now() - t0,
      r2ReadMs, wasmExecMs, cacheHits: cacheHit ? 1 : 0, cacheMisses: cacheHit ? 0 : 1,
    };
  }

  /** Scan only the needed pages from R2 via coalesced Range reads, with cache-before-fetch. */
  private async scanPages(query: QueryDescriptor, meta: TableMeta, t0: number): Promise<QueryResult> {
    let cols = query.projections.length > 0
      ? meta.columns.filter(c => query.projections.includes(c.name))
      : meta.columns;

    if (query.vectorSearch) {
      const vc = query.vectorSearch.column;
      if (!cols.some(c => c.name === vc)) {
        const ec = meta.columns.find(c => c.name === vc);
        if (ec) cols = [...cols, ec];
      }
    }

    // Build per-page ranges, applying page-level skip.
    // Track non-skipped page infos per column so buffer indices stay aligned.
    const ranges: { column: string; offset: number; length: number }[] = [];
    const columnPageInfos = new Map<string, typeof cols[0]["pages"]>();
    let pagesSkipped = 0;
    for (const col of cols) {
      const keptPages: typeof col.pages = [];
      for (const page of col.pages) {
        if (!query.vectorSearch && canSkipPage(page, query.filters, col.name)) { pagesSkipped++; continue; }
        keptPages.push(page);
        ranges.push({ column: col.name, offset: Number(page.byteOffset), length: page.byteLength });
      }
      columnPageInfos.set(col.name, keptPages);
    }

    // 3-tier cache hierarchy:
    //   L1: WASM buffer pool (per-DO, in-memory, fastest, lost on hibernation)
    //   L2: caches.default (per-datacenter, shared across DOs, survives hibernation)
    //   L3: R2 (global, slowest)
    const columnData = new Map<string, ArrayBuffer[]>();
    let cacheHits = 0;   // L1 hits
    let cacheMisses = 0; // L1 misses
    let edgeCacheHits = 0;   // L2 hits
    let edgeCacheMisses = 0; // L2 misses
    const l1Misses: typeof ranges = [];

    // L1: Check WASM buffer pool
    for (const r of ranges) {
      const cacheKey = `${meta.r2Key}:${r.offset}:${r.length}`;
      const cached = this.wasmEngine.cacheGet(cacheKey);
      if (cached) {
        cacheHits++;
        const arr = columnData.get(r.column) ?? [];
        arr.push(cached);
        columnData.set(r.column, arr);
      } else {
        cacheMisses++;
        l1Misses.push(r);
      }
    }

    // L2: caches.default — only for VIP tables (queried >= 3 times).
    // One-off queries against random tables would just pollute the shared cache.
    // VIP tables show repeat access patterns where page caching actually pays off.
    const tableIsVip = this.footerCache.accessCount(query.table) >= VIP_THRESHOLD;
    const uncachedRanges: typeof ranges = [];

    if (tableIsVip && l1Misses.length > 0) {
      const edgeChecks = await Promise.all(
        l1Misses.map(async (r) => {
          const data = await this.edgeCacheGet(meta.r2Key, r.offset, r.length);
          return { range: r, data };
        }),
      );
      for (const { range: r, data } of edgeChecks) {
        if (data) {
          edgeCacheHits++;
          const arr = columnData.get(r.column) ?? [];
          arr.push(data);
          columnData.set(r.column, arr);
          // Promote to L1 for subsequent queries on this DO
          const cacheKey = `${meta.r2Key}:${r.offset}:${r.length}`;
          this.wasmEngine.cacheSet(cacheKey, data);
        } else {
          edgeCacheMisses++;
          uncachedRanges.push(r);
        }
      }
    } else {
      // Not VIP or no L1 misses — go straight to R2 (no L2 lookups, so no L2 misses)
      for (const r of l1Misses) uncachedRanges.push(r);
    }

    // L3: Fetch remaining ranges from R2
    const r2Start = Date.now();
    let bytesRead = 0;

    if (uncachedRanges.length > 0) {
      const coalesced = coalesceRanges(uncachedRanges, autoCoalesceGap(uncachedRanges));

      const fetched = await fetchBounded(
        coalesced.map(c => () =>
          withRetry(() =>
            withTimeout(
              (async () => {
                const obj = await this.r2(meta.r2Key).get(meta.r2Key, { range: { offset: c.offset, length: c.length } });
                return obj ? { ...c, data: await obj.arrayBuffer() } : null;
              })(),
              R2_TIMEOUT_MS,
            ),
          ),
        ),
        8,
      );

      for (const f of fetched) {
        if (!f) continue;
        bytesRead += f.data.byteLength;
        for (const sub of f.ranges) {
          const slice = f.data.slice(sub.offset - f.offset, sub.offset - f.offset + sub.length);
          const arr = columnData.get(sub.column) ?? [];
          arr.push(slice);
          columnData.set(sub.column, arr);

          // Populate L1 cache
          const cacheKey = `${meta.r2Key}:${sub.offset}:${sub.length}`;
          this.wasmEngine.cacheSet(cacheKey, slice);

          // Populate L2 cache (async, fire-and-forget — don't block query)
          // Only for VIP tables — no point caching pages for one-off queries
          if (tableIsVip) {
            this.edgeCachePut(meta.r2Key, sub.offset, sub.length, slice);
          }
        }
      }
    }
    const r2ReadMs = Date.now() - r2Start;

    // IVF-PQ index-aware path: if a vector search is requested, check for an index
    if (query.vectorSearch) {
      const wasmIvfStart = Date.now();
      const indexResult = await this.tryIvfPqSearch(query, meta);
      if (indexResult) {
        indexResult.durationMs = Date.now() - t0;
        indexResult.r2ReadMs = Date.now() - r2Start;
        indexResult.wasmExecMs = Date.now() - wasmIvfStart;
        indexResult.pagesSkipped = pagesSkipped;
        indexResult.cacheHits = cacheHits;
        indexResult.cacheMisses = cacheMisses;
        indexResult.edgeCacheHits = edgeCacheHits;
        indexResult.edgeCacheMisses = edgeCacheMisses;
        return indexResult;
      }
      // Fall through to flat search via SQL executor
    }

    // Parquet path: decode column chunks in JS (handles page headers, dictionary encoding, compression)
    if (meta.format === "parquet") {
      const wasmStart = Date.now();
      const decodedColumns = new Map<string, (number | bigint | string | boolean | null)[]>();
      for (const col of cols) {
        const pages = columnData.get(col.name);
        if (!pages?.length) { decodedColumns.set(col.name, []); continue; }

        // Use non-skipped page infos (aligned with columnData buffers, not col.pages)
        const keptPageInfos = columnPageInfos.get(col.name) ?? col.pages;

        // Concatenate all page buffers for this column (may span multiple row groups)
        const allValues: (number | bigint | string | boolean | null)[] = [];
        for (let pi = 0; pi < pages.length; pi++) {
          const pageInfo = keptPageInfos[pi];
          const encoding = pageInfo?.encoding ?? { compression: "UNCOMPRESSED" };

          // Include dictionary page if present: fetch it from R2 and prepend
          let chunkBuf = pages[pi];
          if (encoding.dictionaryPageOffset !== undefined && encoding.dictionaryPageLength) {
            // Dictionary page is at a different offset — check if it's already included
            // If the fetched range starts at dataPageOffset (not dictionaryPageOffset),
            // we need to fetch the dictionary page separately
            const dictOffset = Number(encoding.dictionaryPageOffset);
            const pageOffset = Number(pageInfo.byteOffset);
            if (dictOffset < pageOffset) {
              // Dictionary page is before data page — try to get from cache or R2
              const dictKey = `${meta.r2Key}:${dictOffset}:${encoding.dictionaryPageLength}`;
              let dictBuf = this.wasmEngine.cacheGet(dictKey);
              if (!dictBuf) {
                const dictObj = await this.r2(meta.r2Key).get(meta.r2Key, {
                  range: { offset: dictOffset, length: encoding.dictionaryPageLength },
                });
                if (dictObj) {
                  dictBuf = await dictObj.arrayBuffer();
                  this.wasmEngine.cacheSet(dictKey, dictBuf);
                  bytesRead += dictBuf.byteLength;
                }
              }
              if (dictBuf) {
                // Prepend dictionary page to data page
                const combined = new Uint8Array(dictBuf.byteLength + chunkBuf.byteLength);
                combined.set(new Uint8Array(dictBuf), 0);
                combined.set(new Uint8Array(chunkBuf), dictBuf.byteLength);
                chunkBuf = combined.buffer;
              }
            }
          }

          const decoded = decodeParquetColumnChunk(
            chunkBuf, encoding, col.dtype, pageInfo?.rowCount ?? 0, this.wasmEngine,
          );
          for (let di = 0; di < decoded.length; di++) allValues.push(decoded[di]);
        }
        decodedColumns.set(col.name, allValues);
      }

      // Try WASM SQL path: register decoded columns → executeQuery (SIMD filter/sort/agg)
      const colNames = cols.map(c => c.name);
      this.wasmEngine.exports.resetHeap();
      let wasmRegistered = true;
      for (const col of cols) {
        const values = decodedColumns.get(col.name);
        if (!values?.length) continue;
        if (!this.wasmEngine.registerDecodedColumn(query.table, col.name, col.dtype, values)) {
          wasmRegistered = false;
          break;
        }
      }

      let rows: Row[];
      if (wasmRegistered) {
        const wasmRows = this.wasmEngine.executeQuery(query);
        this.wasmEngine.clearTable(query.table);
        rows = wasmRows ?? [];
      } else {
        // Fallback to JS path if WASM registration fails (OOM, unsupported dtype)
        rows = this.applyJsPostProcessing(
          this.assembleRowsFromColumns(decodedColumns, colNames), query,
        );
        if (query.aggregates && query.aggregates.length > 0) {
          const partial = computePartialAgg(rows, query);
          rows = finalizePartialAgg(partial, query);
        }
      }

      const wasmExecMs = Date.now() - wasmStart;
      const outputCols = (query.aggregates && query.aggregates.length > 0)
        ? Object.keys(rows[0] ?? {})
        : colNames;
      return {
        rows, rowCount: rows.length, columns: outputCols,
        bytesRead, pagesSkipped, durationMs: Date.now() - t0,
        r2ReadMs, wasmExecMs, cacheHits, cacheMisses,
        edgeCacheHits, edgeCacheMisses,
      };
    }

    // Lance path: zero-copy WASM registration + SQL execution
    const wasmStart = Date.now();
    this.wasmEngine.exports.resetHeap();
    for (const col of cols) {
      const pages = columnData.get(col.name);
      if (!pages?.length) continue;
      const keptPageInfos = columnPageInfos.get(col.name) ?? col.pages;
      if (!this.wasmEngine.registerColumn(query.table, col.name, col.dtype, pages, keptPageInfos, col.listDimension)) {
        throw new Error(`WASM OOM: failed to register column "${col.name}" for table "${query.table}"`);
      }
    }

    const rows = this.wasmEngine.executeQuery(query);
    if (!rows) throw new Error(`WASM query execution failed for table "${query.table}"`);
    this.wasmEngine.clearTable(query.table);
    const wasmExecMs = Date.now() - wasmStart;

    return {
      rows, rowCount: rows.length, columns: cols.map(c => c.name),
      bytesRead, pagesSkipped, durationMs: Date.now() - t0,
      r2ReadMs, wasmExecMs, cacheHits, cacheMisses,
      edgeCacheHits, edgeCacheMisses,
    };
  }

  /**
   * Execute a query using the streaming operator pipeline with R2 spill.
   * Handles ORDER BY without LIMIT, large GROUP BY, etc. on edge.
   */
  private async executeWithPipeline(query: QueryDescriptor, meta: TableMeta, t0: number): Promise<QueryResult> {
    const spill = new R2SpillBackend(this.r2(meta.r2Key), `__spill/${crypto.randomUUID()}`);
    try {
      const scan = new EdgeScanOperator(
        this.r2(meta.r2Key), this.wasmEngine, meta, query, this.footerCache,
        (r2Key, offset, length) => this.edgeCacheGet(r2Key, offset, length),
        (r2Key, offset, length, data) => { this.edgeCachePut(r2Key, offset, length, data); },
      );

      const outputColumns = query.projections.length > 0
        ? query.projections
        : meta.columns.map(c => c.name);

      const pipeline = buildEdgePipeline(scan, query, outputColumns, {
        memoryBudgetBytes: EDGE_MEMORY_BUDGET,
        spill,
      });
      const rows = await drainPipeline(pipeline);

      return {
        rows, rowCount: rows.length, columns: outputColumns,
        bytesRead: scan.bytesRead, pagesSkipped: scan.pagesSkipped,
        durationMs: Date.now() - t0,
        r2ReadMs: scan.r2ReadMs, wasmExecMs: scan.wasmExecMs,
        cacheHits: scan.cacheHits, cacheMisses: scan.cacheMisses,
        edgeCacheHits: scan.edgeCacheHits, edgeCacheMisses: scan.edgeCacheMisses,
        spillBytesWritten: spill.bytesWritten || undefined,
        spillBytesRead: spill.bytesRead || undefined,
      };
    } finally {
      await spill.cleanup();
    }
  }

  /** Execute a join query on edge using operator pipeline with R2 spill. */
  private async executeJoin(query: QueryDescriptor, t0: number): Promise<QueryResult> {
    const join = query.join!;
    const leftR2Key = query.table; // for bucket resolution before meta loads
    const spill = new R2SpillBackend(this.r2(leftR2Key), `__spill/${crypto.randomUUID()}`);

    try {
      // Load left table meta
      let leftMeta: TableMeta | undefined = this.footerCache.get(query.table);
      if (!leftMeta) {
        leftMeta = (await this.loadTableFromR2(query.table)) ?? undefined;
        if (!leftMeta) throw new Error(`Table "${query.table}" not found`);
      }

      // Build left scan (no sort/limit/agg — those apply after join)
      const leftScan = new EdgeScanOperator(
        this.r2(leftMeta.r2Key), this.wasmEngine, leftMeta,
        { ...query, sortColumn: undefined, limit: undefined, offset: undefined, aggregates: undefined, join: undefined },
        this.footerCache,
        (r2Key, offset, length) => this.edgeCacheGet(r2Key, offset, length),
        (r2Key, offset, length, data) => { this.edgeCachePut(r2Key, offset, length, data); },
      );
      let leftPipeline: Operator = leftScan;
      if (query.filters.length > 0 || (query.filterGroups && query.filterGroups.length > 0)) {
        leftPipeline = new FilterOperator(leftPipeline, query.filters, query.filterGroups);
      }

      // Load right table meta
      let rightMeta: TableMeta | undefined = this.footerCache.get(join.right.table);
      if (!rightMeta) {
        rightMeta = (await this.loadTableFromR2(join.right.table)) ?? undefined;
        if (!rightMeta) throw new Error(`Table "${join.right.table}" not found`);
      }

      // Build right scan
      const rightScan = new EdgeScanOperator(
        this.r2(rightMeta.r2Key), this.wasmEngine, rightMeta, join.right,
        this.footerCache,
        (r2Key, offset, length) => this.edgeCacheGet(r2Key, offset, length),
        (r2Key, offset, length, data) => { this.edgeCachePut(r2Key, offset, length, data); },
      );
      let rightPipeline: Operator = rightScan;
      if (join.right.filters.length > 0 || (join.right.filterGroups && join.right.filterGroups.length > 0)) {
        rightPipeline = new FilterOperator(rightPipeline, join.right.filters, join.right.filterGroups);
      }

      // Hash join: right is build side, left is probe side
      let pipeline: Operator = new HashJoinOperator(
        leftPipeline, rightPipeline, join.leftKey, join.rightKey,
        join.type ?? "inner", EDGE_MEMORY_BUDGET, spill,
      );

      // Apply post-join operators
      const outputColumns = query.projections.length > 0
        ? query.projections
        : [...new Set([
            ...leftMeta.columns.map(c => c.name),
            ...rightMeta.columns.map(c => c.name),
          ])];

      pipeline = buildEdgePipeline(pipeline, {
        ...query, filters: [], filterGroups: undefined, join: undefined,
      }, outputColumns, { memoryBudgetBytes: EDGE_MEMORY_BUDGET, spill });

      // The buildEdgePipeline already applies sort/limit/agg/project — but we passed
      // filters: [] since we already filtered above. We also cleared join to prevent recursion.
      // However buildEdgePipeline re-applies FilterOperator if filters > 0, so clearing is correct.

      const rows = await drainPipeline(pipeline);

      return {
        rows, rowCount: rows.length, columns: outputColumns,
        bytesRead: leftScan.bytesRead + rightScan.bytesRead,
        pagesSkipped: leftScan.pagesSkipped + rightScan.pagesSkipped,
        durationMs: Date.now() - t0,
        r2ReadMs: leftScan.r2ReadMs + rightScan.r2ReadMs,
        wasmExecMs: leftScan.wasmExecMs + rightScan.wasmExecMs,
        cacheHits: leftScan.cacheHits + rightScan.cacheHits,
        cacheMisses: leftScan.cacheMisses + rightScan.cacheMisses,
        edgeCacheHits: leftScan.edgeCacheHits + rightScan.edgeCacheHits,
        edgeCacheMisses: leftScan.edgeCacheMisses + rightScan.edgeCacheMisses,
        spillBytesWritten: spill.bytesWritten || undefined,
        spillBytesRead: spill.bytesRead || undefined,
      };
    } finally {
      await spill.cleanup();
    }
  }

  /** Try IVF-PQ indexed vector search. Returns null if no index available. */
  private async tryIvfPqSearch(query: QueryDescriptor, meta: TableMeta): Promise<QueryResult | null> {
    if (!query.vectorSearch) return null;

    // Check for vector index metadata
    const vs = query.vectorSearch;
    const indexInfo = meta.vectorIndexes?.find(
      vi => vi.column === vs.column && vi.type === "ivf_pq",
    );

    // Also try convention-based index path: {r2Key}.index
    const indexPath = indexInfo?.indexPath ?? `${meta.r2Key}.ivf_pq.index`;

    // Check buffer pool cache first
    const cacheKey = `ivf_pq:${indexPath}`;
    let indexData = this.wasmEngine.cacheGet(cacheKey);

    if (!indexData) {
      // Try loading from R2
      const indexObj = await this.r2(indexPath).get(indexPath);
      if (!indexObj) return null; // No index → fall through to flat search

      indexData = await indexObj.arrayBuffer();
      this.wasmEngine.cacheSet(cacheKey, indexData);
    }

    // Load index into WASM
    let handle: number;
    try {
      handle = this.wasmEngine.loadIvfPqIndex(indexData);
    } catch {
      return null; // Invalid index format → fall through
    }

    try {
      const nprobe = indexInfo?.config?.nProbe ?? 10;
      const { indices, scores } = this.wasmEngine.searchIvfPq(handle, vs.queryVector, vs.topK, nprobe);

      // Build result rows with distances
      const rows: Row[] = [];
      for (let i = 0; i < indices.length; i++) {
        rows.push({
          _index: indices[i],
          _distance: scores[i],
          _score: 1 / (1 + scores[i]), // Convert L2 distance to similarity score
        });
      }

      this.log("info", "ivf_pq_search", {
        table: query.table, column: vs.column, topK: vs.topK,
        nprobe, resultsFound: rows.length,
      });

      return {
        rows,
        rowCount: rows.length,
        columns: ["_index", "_distance", "_score"],
        bytesRead: indexData.byteLength,
        pagesSkipped: 0,
        durationMs: 0,
      };
    } finally {
      this.wasmEngine.freeIvfPqIndex(handle);
    }
  }


  private async readColumnMeta(r2Key: string, footer: Footer): Promise<ColumnMeta[]> {
    const len = Number(footer.columnMetaOffsetsStart) - Number(footer.columnMetaStart);
    if (len <= 0) return [];
    const obj = await this.r2(r2Key).get(r2Key, { range: { offset: Number(footer.columnMetaStart), length: len } });
    if (!obj) return [];
    return parseColumnMetaFromProtobuf(await obj.arrayBuffer(), footer.numColumns);
  }

  private async loadTableFromR2(tableName: string): Promise<TableMeta | null> {
    const candidates = [
      `${tableName}.lance`, `${tableName}.parquet`, tableName,
      `data/${tableName}.lance`, `data/${tableName}.parquet`, `data/${tableName}`,
    ];
    for (const r2Key of candidates) {
      const head = await this.r2(r2Key).head(r2Key);
      if (!head) continue;

      const fileSize = BigInt(head.size);
      const tailSize = Math.min(Number(fileSize), 40);
      const obj = await this.r2(r2Key).get(r2Key, { range: { offset: Number(fileSize) - tailSize, length: tailSize } });
      if (!obj) continue;

      const tailBuf = await obj.arrayBuffer();
      const fmt = detectFormat(tailBuf);

      if (fmt === "parquet") {
        const footerLen = getParquetFooterLength(tailBuf);
        if (!footerLen) continue;

        const footerOffset = Number(fileSize) - footerLen - 8;
        const footerObj = await this.r2(r2Key).get(r2Key, { range: { offset: footerOffset, length: footerLen } });
        if (!footerObj) continue;

        const parquetMeta = parseParquetFooter(await footerObj.arrayBuffer());
        if (!parquetMeta) continue;

        const meta = parquetMetaToTableMeta(parquetMeta, r2Key, fileSize);
        meta.name = tableName;
        this.footerCache.set(tableName, meta);
        await this.ctx.storage.put(`table:${tableName}`, meta);
        return meta;
      }

      if (fmt === "lance") {
        const footer = parseFooter(tailBuf);
        if (!footer) continue;

        const columns = await this.readColumnMeta(r2Key, footer);
        const meta: TableMeta = {
          name: tableName, footer, format: "lance", columns,
          totalRows: columns[0]?.pages.reduce((s, p) => s + p.rowCount, 0) ?? 0,
          fileSize, r2Key, updatedAt: Date.now(),
        };
        this.footerCache.set(tableName, meta);
        await this.ctx.storage.put(`table:${tableName}`, meta);
        return meta;
      }
    }

    // Try as Lance dataset directory (has _versions/ with manifests)
    const datasetMeta = await this.loadDatasetFromR2(tableName);
    if (datasetMeta) return datasetMeta.fragmentMetas.values().next().value ?? null;

    // Try as Iceberg table (has metadata/ directory)
    const icebergMeta = await this.loadIcebergFromR2(tableName);
    if (icebergMeta) return icebergMeta.fragmentMetas.values().next().value ?? null;

    return null;
  }

  /** Discover a multi-fragment Lance dataset in R2 by listing _versions/ manifests. */
  private async loadDatasetFromR2(tableName: string): Promise<DatasetMeta | null> {
    for (const prefix of [`${tableName}.lance/`, `${tableName}/`, `data/${tableName}.lance/`, `data/${tableName}/`]) {
      const listed = await this.r2(prefix).list({ prefix: `${prefix}_versions/`, limit: 100 });
      const manifestKeys = listed.objects
        .filter(o => o.key.endsWith(".manifest"))
        .sort((a, b) => a.key.localeCompare(b.key));
      if (manifestKeys.length === 0) continue;

      // Read latest manifest
      const latestKey = manifestKeys[manifestKeys.length - 1].key;
      const manifestObj = await this.r2(latestKey).get(latestKey);
      if (!manifestObj) continue;

      const manifest = parseManifest(await manifestObj.arrayBuffer());
      if (!manifest) continue;

      // Read footer + columns for each fragment
      const fragmentMetas = new Map<number, TableMeta>();
      for (const frag of manifest.fragments) {
        // Try filePath as-is first, then with data/ prefix (Lance stores relative paths without data/)
        const candidates = [
          `${prefix}${frag.filePath}`,
          `${prefix}data/${frag.filePath}`,
        ];
        let head: { size: number } | null = null;
        let fragKey = candidates[0];
        for (const candidate of candidates) {
          head = await this.r2(candidate).head(candidate);
          if (head) { fragKey = candidate; break; }
        }
        if (!head) continue;

        const fileSize = BigInt(head.size);
        const footerObj = await this.r2(fragKey).get(fragKey, { range: { offset: Number(fileSize) - 40, length: 40 } });
        if (!footerObj) continue;

        const footer = parseFooter(await footerObj.arrayBuffer());
        if (!footer) continue;

        let columns = await this.readColumnMeta(fragKey, footer);
        // Apply manifest schema names/types to columns (v2 protobuf may have encoding paths instead of names)
        if (manifest.schema.length > 0) {
          const leafFields = manifest.schema.filter(f => f.parentId === -1 || f.parentId === 0);
          columns = columns.map((col, i) => {
            const field = leafFields[i];
            if (!field) return col;
            return { ...col, name: field.name, dtype: logicalTypeToDataType(field.logicalType) };
          });
        }
        fragmentMetas.set(frag.id, {
          name: frag.filePath, footer, format: "lance", columns,
          totalRows: frag.physicalRows,
          fileSize, r2Key: fragKey, updatedAt: Date.now(),
        });
      }

      if (fragmentMetas.size === 0) continue;

      const dataset: DatasetMeta = {
        name: tableName, r2Prefix: prefix, manifest,
        fragmentMetas, totalRows: manifest.totalRows, updatedAt: Date.now(),
      };
      this.datasetCache.set(tableName, dataset);
      this.buildPartitionCatalog(tableName, fragmentMetas);
      this.evictDatasetCache();
      return dataset;
    }
    return null;
  }

  /** Load Parquet file metadata from R2 (head → tail → footer → TableMeta). Returns null if file not found or invalid. */
  private async loadParquetR2Meta(r2Key: string): Promise<TableMeta | null> {
    const head = await this.r2(r2Key).head(r2Key);
    if (!head) return null;
    const fileSize = BigInt(head.size);
    const tailObj = await this.r2(r2Key).get(r2Key, {
      range: { offset: Math.max(0, Number(fileSize) - 8), length: Math.min(8, Number(fileSize)) },
    });
    if (!tailObj) return null;
    const footerLen = getParquetFooterLength(await tailObj.arrayBuffer());
    if (!footerLen) return null;
    const footerObj = await this.r2(r2Key).get(r2Key, {
      range: { offset: Number(fileSize) - footerLen - 8, length: footerLen },
    });
    if (!footerObj) return null;
    const parquetMeta = parseParquetFooter(await footerObj.arrayBuffer());
    if (!parquetMeta) return null;
    return parquetMetaToTableMeta(parquetMeta, r2Key, fileSize);
  }

  /** Load Parquet fragment metadata for a list of paths, returning (fragmentMetas, totalRows). */
  private async loadParquetFragments(parquetPaths: string[], prefix: string): Promise<{ fragmentMetas: Map<number, TableMeta>; totalRows: number }> {
    const fragmentMetas = new Map<number, TableMeta>();
    let totalRows = 0;
    for (let i = 0; i < parquetPaths.length; i++) {
      const parquetKey = parquetPaths[i].startsWith(prefix) ? parquetPaths[i] : `${prefix}${parquetPaths[i]}`;
      const meta = await this.loadParquetR2Meta(parquetKey);
      if (!meta) continue;
      meta.name = parquetPaths[i];
      fragmentMetas.set(i, meta);
      totalRows += meta.totalRows;
    }
    return { fragmentMetas, totalRows };
  }

  /** Cache an Iceberg dataset in both datasetCache (for multi-fragment queries) and return the metadata. */
  private cacheIcebergDataset(
    tableName: string, prefix: string, icebergMeta: { schema: IcebergDatasetMeta["schema"]; currentSnapshotId: string },
    parquetPaths: string[], fragmentMetas: Map<number, TableMeta>, totalRows: number,
  ): IcebergDatasetMeta {
    const dataset: IcebergDatasetMeta = {
      name: tableName, r2Prefix: prefix,
      schema: icebergMeta.schema, snapshotId: icebergMeta.currentSnapshotId,
      parquetFiles: parquetPaths, fragmentMetas, totalRows, updatedAt: Date.now(),
    };
    this.datasetCache.set(tableName, {
      name: tableName, r2Prefix: prefix,
      manifest: { version: 0, fragments: parquetPaths.map((p, idx) => ({ id: idx, filePath: p, physicalRows: 0 })), totalRows, schema: [] },
      fragmentMetas, totalRows, updatedAt: Date.now(),
    });
    this.buildPartitionCatalog(tableName, fragmentMetas);
    this.evictDatasetCache();
    return dataset;
  }

  /** Load an Iceberg table from an explicit metadata.json key (no R2 list() needed). */
  private async loadIcebergByKey(tableName: string, metadataKey: string): Promise<IcebergDatasetMeta | null> {
    const metaObj = await this.r2(metadataKey).get(metadataKey);
    if (!metaObj) return null;

    const metaJson = await metaObj.text();
    const icebergMeta = parseIcebergMetadata(metaJson);
    if (!icebergMeta) return null;

    // Derive prefix from metadataKey (e.g., "bench_iceberg_100k/metadata/v1.metadata.json" → "bench_iceberg_100k/")
    const prefix = metadataKey.replace(/metadata\/.*$/, "");

    const manifestListKey = `${prefix}${icebergMeta.manifestListPath}`;
    const manifestListObj = await this.r2(manifestListKey).get(manifestListKey);
    if (!manifestListObj) return null;

    const manifestListBytes = await manifestListObj.arrayBuffer();
    const parquetPaths = extractParquetPathsFromManifest(manifestListBytes);
    if (parquetPaths.length === 0) return null;

    const { fragmentMetas, totalRows } = await this.loadParquetFragments(parquetPaths, prefix);
    if (fragmentMetas.size === 0) return null;

    return this.cacheIcebergDataset(tableName, prefix, icebergMeta, parquetPaths, fragmentMetas, totalRows);
  }

  /** Discover an Iceberg table in R2 by listing metadata/ for .metadata.json files. */
  private async loadIcebergFromR2(tableName: string): Promise<IcebergDatasetMeta | null> {
    for (const prefix of [`${tableName}/`, `data/${tableName}/`]) {
      const listed = await this.r2(prefix).list({ prefix: `${prefix}metadata/`, limit: 100 });
      const metadataKeys = listed.objects
        .filter(o => o.key.endsWith(".metadata.json"))
        .sort((a, b) => a.key.localeCompare(b.key));
      if (metadataKeys.length === 0) continue;

      const latestKey = metadataKeys[metadataKeys.length - 1].key;
      const metaObj = await this.r2(latestKey).get(latestKey);
      if (!metaObj) continue;

      const icebergMeta = parseIcebergMetadata(await metaObj.text());
      if (!icebergMeta) continue;

      const manifestListKey = `${prefix}${icebergMeta.manifestListPath}`;
      const manifestListObj = await this.r2(manifestListKey).get(manifestListKey);
      if (!manifestListObj) continue;

      const parquetPaths = extractParquetPathsFromManifest(await manifestListObj.arrayBuffer());
      if (parquetPaths.length === 0) continue;

      const { fragmentMetas, totalRows } = await this.loadParquetFragments(parquetPaths, prefix);
      if (fragmentMetas.size === 0) continue;

      return this.cacheIcebergDataset(tableName, prefix, icebergMeta, parquetPaths, fragmentMetas, totalRows);
    }
    return null;
  }

  /** Execute a query across multiple fragments of a dataset. */
  private async executeMultiFragment(query: QueryDescriptor, dataset: DatasetMeta, t0: number): Promise<QueryResult> {
    // Phase 1: Partition catalog pruning (O(1) lookup by partition key)
    let candidateFragmentIds: Set<number> | null = null;
    if (query.filters.length > 0) {
      // Try each known partition catalog for this dataset
      const catalog = this.partitionCatalogs.get(dataset.name);
      if (catalog) {
        const pruned = catalog.prune(query.filters);
        if (pruned !== null) {
          candidateFragmentIds = new Set(pruned);
          this.log("info", "partition_catalog_prune", {
            column: catalog.column,
            total: dataset.fragmentMetas.size,
            afterCatalog: candidateFragmentIds.size,
          });
        }
      }
    }

    // Phase 2: Fragment-level min/max pruning on remaining candidates
    const allFragments = candidateFragmentIds
      ? [...dataset.fragmentMetas.entries()]
          .filter(([id]) => candidateFragmentIds!.has(id))
          .map(([, meta]) => meta)
      : [...dataset.fragmentMetas.values()];

    const fragments = allFragments.filter(meta =>
      !canSkipFragment(meta, query.filters, query.filterGroups),
    );
    const fragmentsSkipped = (candidateFragmentIds
      ? dataset.fragmentMetas.size
      : allFragments.length) - fragments.length;
    if (fragmentsSkipped > 0) {
      this.log("info", "fragment_prune", {
        total: allFragments.length, skipped: fragmentsSkipped, remaining: fragments.length,
      });
    }

    if (fragments.length === 0) {
      // All fragments pruned — return empty result
      return {
        rows: [], rowCount: 0, columns: [],
        bytesRead: 0, pagesSkipped: 0, durationMs: Date.now() - t0,
        r2ReadMs: 0, wasmExecMs: 0, cacheHits: 0, cacheMisses: 0,
        edgeCacheHits: 0, edgeCacheMisses: 0,
      };
    }

    // Large datasets (>4 fragments): fan out to Fragment DOs
    if (fragments.length > 4) {
      return this.executeWithFragmentDOs(query, fragments, t0);
    }

    // For unbounded queries or aggregation, use per-fragment pipeline with spill.
    // Aggregation must scan all rows before LIMIT applies to the output.
    const hasLimit = query.limit !== undefined;
    const hasAgg = query.aggregates && query.aggregates.length > 0;
    if ((!hasLimit || hasAgg) && !query.vectorSearch) {
      // Build a pipeline per fragment using executeWithPipeline, then merge
      const partials: QueryResult[] = [];
      for (const meta of fragments) {
        partials.push(await this.executeWithPipeline(query, meta, t0));
      }
      return mergeQueryResults(partials, query);
    }

    // Bounded queries: safe to materialize per-fragment
    const partials: QueryResult[] = [];
    for (const meta of fragments) {
      const hasPages = meta.columns.some(c => c.pages.length > 0);
      if (!hasPages && meta.format === "lance" && meta.r2Key) {
        partials.push(await this.executeLanceWholeFile(query, meta, t0));
      } else {
        partials.push(await this.scanPages(query, meta, t0));
      }
    }
    return mergeQueryResults(partials, query);
  }

  /** Claim N available slots from the Fragment DO pool. Returns slot indices.
   *  Slots are per-datacenter, named frag-{region}-slot-{N}.
   *  Idle slots cost nothing (hibernated DOs). Active tracking prevents
   *  concurrent queries from queueing behind each other on the same slot. */
  private claimSlots(needed: number): number[] {
    const slots: number[] = [];
    for (let i = 0; i < FRAGMENT_POOL_MAX && slots.length < needed; i++) {
      if (!this.activeFragmentSlots.has(i)) slots.push(i);
    }
    for (const s of slots) this.activeFragmentSlots.add(s);
    return slots; // may be fewer than requested — caller handles overflow
  }

  private releaseSlots(slots: number[]): void {
    for (const s of slots) this.activeFragmentSlots.delete(s);
  }

  /** Fan-out query to pooled Fragment DOs for parallel scanning.
   *  Uses Promise.allSettled for partial failure tolerance.
   *  Pool is per-datacenter (frag-{region}-slot-{N}), max FRAGMENT_POOL_MAX slots. */
  private async executeWithFragmentDOs(query: QueryDescriptor, prunedFragments: TableMeta[], t0: number): Promise<QueryResult> {
    const allFragItems = prunedFragments.map(meta => ({ r2Key: meta.r2Key, meta }));

    // Claim as many slots as we can (may be fewer than fragments)
    const slotsWanted = Math.min(allFragItems.length, FRAGMENT_POOL_MAX);
    const slots = this.claimSlots(slotsWanted);

    const region = (await this.ctx.storage.get<string>("region")) ?? "default";

    try {
      if (slots.length === 0) {
        // All slots busy — scan locally in parallel batches
        this.log("info", "fragment_fan_out_local", { fragments: allFragItems.length, reason: "no_slots" });
        const partials = await Promise.all(
          allFragItems.map(({ meta }) => this.scanPages(query, meta, t0)),
        );
        return mergeQueryResults(partials, query);
      }

      // Distribute fragments across available slots
      const chunkSize = Math.ceil(allFragItems.length / slots.length);
      const groups: typeof allFragItems[] = [];
      for (let i = 0; i < allFragItems.length; i += chunkSize) {
        groups.push(allFragItems.slice(i, i + chunkSize));
      }

      this.log("info", "fragment_fan_out", {
        fragments: allFragItems.length, slots: slots.length,
        chunksPerSlot: chunkSize,
      });

      const settled = await Promise.allSettled(groups.map(async (group, idx) => {
        if (idx >= slots.length) {
          // Overflow: scan locally (shouldn't happen with Math.ceil, but safe)
          const partials = await Promise.all(
            group.map(({ meta }) => this.scanPages(query, meta, t0)),
          );
          return mergeQueryResults(partials, query);
        }
        const doName = `frag-${region}-slot-${slots[idx]}`;
        const doId = this.env.FRAGMENT_DO.idFromName(doName);
        const fragmentRpc = this.env.FRAGMENT_DO.get(doId) as unknown as { scanRpc(fragments: typeof group, query: QueryDescriptor): Promise<QueryResult> };

        return withTimeout(
          fragmentRpc.scanRpc(group, query),
          FRAGMENT_TIMEOUT_MS,
        );
      }));

      const fulfilled: QueryResult[] = [];
      for (const s of settled) {
        if (s.status === "fulfilled") {
          fulfilled.push(s.value);
        } else {
          this.log("warn", "fragment_do_failed", { reason: String(s.reason) });
        }
      }

      if (fulfilled.length === 0) {
        throw new Error("All Fragment DOs failed");
      }

      const merged = mergeQueryResults(fulfilled, query);
      merged.durationMs = Date.now() - t0;
      return merged;
    } finally {
      this.releaseSlots(slots);
    }
  }

  // ── RPC methods ────────────────────────────────────────────────────────

  /** Common RPC preamble — ensure WASM + footer cache is warm. */
  private async ensureReady(): Promise<void> {
    await this.ensureInitialized();
    if (!this.registeredWithMaster) this.registerWithMaster();
  }

  /** Parse descriptor and ensure ready — shared by all RPC methods. */
  private async rpcParseQuery(descriptor: unknown): Promise<QueryDescriptor> {
    await this.ensureReady();
    return this.parseQuery(descriptor);
  }

  async queryRpc(descriptor: unknown): Promise<QueryResult> {
    return this.executeQuery(await this.rpcParseQuery(descriptor));
  }

  async countRpc(descriptor: unknown): Promise<number> {
    return this.executeCount(await this.rpcParseQuery(descriptor));
  }

  async existsRpc(descriptor: unknown): Promise<boolean> {
    const query = await this.rpcParseQuery(descriptor);
    query.limit = 1;
    return (await this.executeQuery(query)).rowCount > 0;
  }

  async firstRpc(descriptor: unknown): Promise<Row | null> {
    const query = await this.rpcParseQuery(descriptor);
    query.limit = 1;
    return (await this.executeQuery(query)).rows[0] ?? null;
  }

  async explainRpc(descriptor: unknown): Promise<ExplainResult> {
    return this.executeExplain(await this.rpcParseQuery(descriptor));
  }

  async streamRpc(descriptor: unknown): Promise<ReadableStream<Uint8Array>> {
    const result = await this.executeQuery(await this.rpcParseQuery(descriptor));
    const rows = result.rows;
    const STREAM_BATCH_SIZE = 4096;
    let idx = 0;
    return new ReadableStream<Uint8Array>({
      pull(controller) {
        if (idx >= rows.length) { controller.close(); return; }
        const end = Math.min(idx + STREAM_BATCH_SIZE, rows.length);
        const batch = rows.slice(idx, end);
        idx = end;
        const buf = encodeColumnarRun(batch);
        // Length-prefix: 4 bytes LE uint32 + payload
        const frame = new Uint8Array(4 + buf.byteLength);
        new DataView(frame.buffer).setUint32(0, buf.byteLength, true);
        frame.set(new Uint8Array(buf), 4);
        controller.enqueue(frame);
      },
    });
  }

  async invalidateRpc(payload: unknown): Promise<void> {
    await this.ensureReady();
    await this.executeInvalidation(payload as Parameters<typeof this.executeInvalidation>[0]);
  }

  async listTablesRpc(): Promise<{ tables: unknown[] }> {
    await this.ensureReady();
    const tables = [...this.footerCache.entries()].map(([name, entry]) => ({
      name, columns: entry.value.columns.map(c => c.name), totalRows: entry.value.totalRows,
      updatedAt: entry.value.updatedAt, accessCount: entry.accessCount,
    }));
    return { tables };
  }

  async getMetaRpc(table: string): Promise<TableMeta | null> {
    await this.ensureReady();
    return this.footerCache.get(table) ?? null;
  }

  async diagnosticsRpc(): Promise<Record<string, unknown>> {
    await this.ensureReady();
    const cache = this.wasmEngine.cacheStats();
    return {
      tableCount: this.footerCache.size,
      datasetCount: this.datasetCache.size,
      tables: [...this.footerCache.entries()].map(([name, entry]) => ({
        name, totalRows: entry.value.totalRows, columns: entry.value.columns.length,
        updatedAt: entry.value.updatedAt, accessCount: entry.accessCount,
        isVip: entry.accessCount >= VIP_THRESHOLD,
      })),
      footerCacheStats: this.footerCache.stats(),
      cache: {
        entries: cache.count,
        bytesUsed: cache.bytes,
        maxBytes: cache.maxBytes,
        utilizationPct: cache.maxBytes > 0 ? Math.round((cache.bytes / cache.maxBytes) * 100) : 0,
      },
      wasm: {
        memoryBytes: this.wasmEngine.exports.memory.buffer.byteLength,
        memoryMB: Math.round(this.wasmEngine.exports.memory.buffer.byteLength / (1024 * 1024)),
      },
      activeFragmentSlots: this.activeFragmentSlots.size,
      partitionCatalogs: [...this.partitionCatalogs.entries()].map(([name, cat]) => ({
        table: name, ...cat.stats(),
      })),
    };
  }

  async registerIcebergRpc(body: unknown): Promise<unknown> {
    await this.ensureReady();
    const { table, metadataKey } = body as { table: string; metadataKey: string };
    if (!table || !metadataKey) throw new Error("Missing table or metadataKey");

    const result = await this.loadIcebergByKey(table, metadataKey);
    if (!result) throw new Error("Failed to load Iceberg metadata");
    return { registered: true, table, totalRows: result.totalRows, files: result.parquetFiles.length };
  }

  /** Store the region name when called via Worker. */
  async setRegion(region: string): Promise<void> {
    const stored = await this.ctx.storage.get<string>("region");
    if (!stored) await this.ctx.storage.put("region", region);
  }

  // ── Edge page cache (caches.default — L2, per-datacenter, shared) ────

  private edgeCacheUrl(r2Key: string, offset: number, length: number): string {
    return `https://querymode-pages/${encodeURIComponent(r2Key)}/${offset}/${length}`;
  }

  private async edgeCacheGet(r2Key: string, offset: number, length: number): Promise<ArrayBuffer | null> {
    try {
      const resp = await (caches as unknown as { default: Cache }).default.match(
        new Request(this.edgeCacheUrl(r2Key, offset, length)),
      );
      return resp ? resp.arrayBuffer() : null;
    } catch {
      return null;
    }
  }

  private async edgeCachePut(r2Key: string, offset: number, length: number, data: ArrayBuffer): Promise<void> {
    try {
      await (caches as unknown as { default: Cache }).default.put(
        new Request(this.edgeCacheUrl(r2Key, offset, length)),
        new Response(data, {
          headers: {
            "Cache-Control": "public, max-age=604800",
            "Content-Type": "application/octet-stream",
            "Content-Length": String(data.byteLength),
          },
        }),
      );
    } catch {
      // Fire-and-forget — R2 is source of truth
    }
  }

}
