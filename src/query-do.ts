import type { ColumnMeta, DataType, Env, ExplainResult, Footer, Row, TableMeta, DatasetMeta, IcebergDatasetMeta, QueryResult } from "./types.js";
import type { QueryDescriptor } from "./client.js";
import { parseFooter, parseColumnMetaFromProtobuf } from "./footer.js";
import { parseManifest, logicalTypeToDataType } from "./manifest.js";
import { detectFormat, getParquetFooterLength, parseParquetFooter, parquetMetaToTableMeta } from "./parquet.js";
import { parseIcebergMetadata, extractParquetPathsFromManifest } from "./iceberg.js";
import { canSkipPage, bigIntReplacer, assembleRows } from "./decode.js";
import { decodeParquetColumnChunk } from "./parquet-decode.js";
import { instantiateWasm, type WasmEngine } from "./wasm-engine.js";
import { mergeQueryResults } from "./merge.js";
import { coalesceRanges, fetchBounded, withRetry, withTimeout } from "./coalesce.js";
import { VipCache } from "./vip-cache.js";
import wasmModule from "./wasm-module.js";

const FRAGMENT_POOL_MAX = 20; // Max Fragment DO slots per datacenter (idle slots cost nothing)
const R2_TIMEOUT_MS = 10_000;
const FRAGMENT_TIMEOUT_MS = 25_000;
const FOOTER_CACHE_MAX = 1000; // ~4KB per footer = ~4MB at capacity
const VIP_THRESHOLD = 3; // Accesses needed to become "VIP" (protected from eviction)
const DATASET_CACHE_MAX = 100; // Max cached datasets before eviction
const RESULT_CACHE_MAX = 200; // Max cached query results
const RESULT_VIP_THRESHOLD = 2; // Accesses needed for VIP result cache

/**
 * Query DO — per-region reader with cached footers.
 * WASM (Zig SIMD) handles all compute; TS handles I/O orchestration only.
 */
export class QueryDO implements DurableObject {
  private state: DurableObjectState;
  private env: Env;
  private footerCache = new VipCache<string, TableMeta>(FOOTER_CACHE_MAX, VIP_THRESHOLD);
  private datasetCache = new Map<string, DatasetMeta>();
  private resultCache = new VipCache<string, QueryResult>(RESULT_CACHE_MAX, RESULT_VIP_THRESHOLD);
  private wasmEngine!: WasmEngine;
  private activeFragmentSlots = new Set<number>(); // slots currently scanning
  private initialized = false;
  private registeredWithMaster = false;

  constructor(state: DurableObjectState, env: Env) {
    this.state = state;
    this.env = env;
  }

  private log(level: "info" | "warn" | "error", msg: string, data?: Record<string, unknown>): void {
    console[level === "error" ? "error" : level === "warn" ? "warn" : "log"](
      JSON.stringify({ ts: new Date().toISOString(), level, msg, ...data }),
    );
  }

  /** Evict oldest dataset entry (by updatedAt) when cache exceeds max size. */
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

  private json(body: unknown, status = 200): Response {
    return new Response(JSON.stringify(body, bigIntReplacer), {
      status, headers: { "content-type": "application/json" },
    });
  }

  async fetch(request: Request): Promise<Response> {
    const region = request.headers.get("x-querymode-region");
    if (region) {
      const stored = await this.state.storage.get<string>("region");
      if (!stored) await this.state.storage.put("region", region);
    }
    await this.ensureInitialized();
    if (!this.registeredWithMaster) this.registerWithMaster();

    switch (new URL(request.url).pathname) {
      case "/invalidate":   return this.handleInvalidation(request);
      case "/query":        return this.handleQuery(request);
      case "/query/stream": return this.handleQueryStream(request);
      case "/query/count":  return this.handleCount(request);
      case "/query/exists": return this.handleExists(request);
      case "/query/first":  return this.handleFirst(request);
      case "/query/explain": return this.handleExplain(request);
      case "/tables":       return this.handleListTables();
      case "/meta":         return this.handleGetMeta(request);
      case "/diagnostics":  return this.handleDiagnostics();
      case "/register-iceberg": return this.handleRegisterIceberg(request);
      default:              return new Response("Not found", { status: 404 });
    }
  }

  private async ensureInitialized(): Promise<void> {
    if (this.initialized) return;
    this.initialized = true;

    const stored = await this.state.storage.list<TableMeta>({ prefix: "table:" });
    for (const [key, meta] of stored) this.footerCache.set(key.replace("table:", ""), meta);

    this.wasmEngine = await instantiateWasm(wasmModule);

    // Register with Master for invalidation broadcasts
    this.registerWithMaster();
  }

  private registerWithMaster(): void {
    const master = this.env.MASTER_DO.get(this.env.MASTER_DO.idFromName("master"));
    this.state.storage.get<string>("region").then(async (region) => {
      try {
        const resp = await master.fetch(new Request("http://internal/register", {
          method: "POST",
          body: JSON.stringify({ queryDoId: this.state.id.toString(), region: region ?? "unknown" }),
          headers: { "content-type": "application/json" },
        }));
        this.registeredWithMaster = true;
        // Master returns current table timestamps — refresh any stale entries
        // This closes the gap where broadcasts were missed during hibernation
        const body = await resp.json() as {
          registered: boolean; region: string;
          tableVersions?: Record<string, { r2Key: string; updatedAt: number }>;
        };
        if (body.tableVersions) {
          await this.refreshStaleTables(body.tableVersions);
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
      const head = await this.env.DATA_BUCKET.head(r2Key);
      if (!head) continue;

      const fileSize = BigInt(head.size);
      const tailSize = Math.min(Number(fileSize), 40);
      const obj = await this.env.DATA_BUCKET.get(r2Key, { range: { offset: Number(fileSize) - tailSize, length: tailSize } });
      if (!obj) continue;

      const tailBuf = await obj.arrayBuffer();
      const fmt = detectFormat(tailBuf);

      let meta: TableMeta;
      if (fmt === "parquet") {
        const footerLen = getParquetFooterLength(tailBuf);
        if (!footerLen) continue;
        const footerObj = await this.env.DATA_BUCKET.get(r2Key, {
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
      await this.state.storage.put(`table:${table}`, meta);
    }
  }

  private async handleInvalidation(request: Request): Promise<Response> {
    const body = (await request.json()) as {
      table: string; r2Key: string; footerBytes: number[];
      columns?: ColumnMeta[]; fileSize?: string; timestamp: number;
      format?: "lance" | "parquet" | "iceberg";
      totalRows?: number; r2Prefix?: string;
    };

    // If broadcast includes r2Prefix, this is a dataset — lazy-load full dataset from R2
    if (body.r2Prefix) {
      this.datasetCache.delete(body.table);
      this.resultCache.invalidateByPrefix(`qr:${body.table}:`);
      this.wasmEngine.clearTable(body.table);
      this.wasmEngine.cacheClear();
      // Trigger lazy-load of the full dataset (reads manifest + all fragments)
      const dataset = await this.loadDatasetFromR2(body.table);
      if (dataset) {
        return this.json({ updated: true, table: body.table, fragments: dataset.fragmentMetas.size });
      }
      // Fall through to single-fragment cache if dataset load fails
    }

    const fmt = body.format ?? "lance";
    let columns: ColumnMeta[];
    let footer: Footer | undefined;

    if (fmt === "parquet" && body.columns) {
      // Parquet invalidation — columns already parsed by Master
      columns = body.columns;
    } else {
      // Lance invalidation — parse footer from raw bytes
      const parsed = parseFooter(new Uint8Array(body.footerBytes).buffer);
      if (!parsed) return this.json({ error: "Invalid footer" }, 400);
      footer = parsed;
      columns = body.columns ?? await this.readColumnMeta(body.r2Key, parsed);
    }

    // Use totalRows from broadcast if available and > 0, otherwise fall back to column computation
    const computedRows = columns[0]?.pages.reduce((s, p) => s + p.rowCount, 0) ?? 0;
    const totalRows = (body.totalRows != null && body.totalRows > 0) ? body.totalRows : computedRows;

    const meta: TableMeta = {
      name: body.table, footer, format: fmt, columns,
      totalRows,
      fileSize: body.fileSize ? BigInt(body.fileSize) : 0n,
      r2Key: body.r2Key, updatedAt: body.timestamp,
    };

    this.footerCache.set(body.table, meta);
    this.resultCache.invalidateByPrefix(`qr:${body.table}:`);
    this.wasmEngine.clearTable(body.table);
    this.wasmEngine.cacheClear();
    await this.state.storage.put(`table:${body.table}`, meta);
    return this.json({ updated: true, table: body.table });
  }

  private async handleQuery(request: Request): Promise<Response> {
    const requestId = request.headers.get("x-querymode-request-id") ?? crypto.randomUUID();
    const body = await request.json() as Record<string, unknown>;
    if (!body.table || typeof body.table !== "string") {
      return this.json({ error: "Missing or invalid 'table' field" }, 400);
    }
    const query = this.parseQuery(body);
    try {
      const result = await this.executeQuery(query);
      result.requestId = requestId;
      this.log("info", "query_complete", {
        requestId, table: query.table, rowCount: result.rowCount,
        bytesRead: result.bytesRead, durationMs: result.durationMs,
        r2ReadMs: result.r2ReadMs, wasmExecMs: result.wasmExecMs,
        cacheHits: result.cacheHits, cacheMisses: result.cacheMisses,
      });
      return this.json(result);
    } catch (err) {
      const msg = err instanceof Error ? `${err.message}\n${err.stack}` : String(err);
      this.log("error", "query_error", {
        requestId, table: query.table, error: msg,
        filterCount: query.filters.length,
        projectionCount: query.projections.length,
      });
      return this.json({ error: msg }, 500);
    }
  }

  private queryKey(query: QueryDescriptor): string {
    const normalized = {
      t: query.table,
      f: [...query.filters].sort((a, b) => a.column.localeCompare(b.column) || a.op.localeCompare(b.op)),
      p: [...query.projections].sort(),
      s: query.sortColumn, sd: query.sortDirection,
      l: query.limit,
      a: query.aggregates, g: query.groupBy,
    };
    const str = JSON.stringify(normalized, bigIntReplacer);
    // FNV-1a hash
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return `qr:${query.table}:${(h >>> 0).toString(36)}`;
  }

  private parseQuery(request_body: Record<string, unknown>): QueryDescriptor {
    return {
      table: request_body.table as string,
      filters: (request_body.filters ?? []) as QueryDescriptor["filters"],
      projections: (request_body.projections ?? request_body.select ?? []) as string[],
      sortColumn: request_body.sortColumn as string | undefined,
      sortDirection: request_body.sortDirection as "asc" | "desc" | undefined,
      limit: request_body.limit as number | undefined,
      vectorSearch: request_body.vectorSearch as QueryDescriptor["vectorSearch"],
      aggregates: request_body.aggregates as QueryDescriptor["aggregates"],
      groupBy: request_body.groupBy as string[] | undefined,
      cacheTTL: request_body.cacheTTL as number | undefined,
    };
  }

  private async handleCount(request: Request): Promise<Response> {
    const body = await request.json() as Record<string, unknown>;
    if (!body.table || typeof body.table !== "string") return this.json({ error: "Missing 'table'" }, 400);
    const query = this.parseQuery(body);
    try {
      // Fast path: no filters — sum page rowCounts from cached metadata
      if (query.filters.length === 0) {
        const meta = this.footerCache.get(query.table)
          ?? (await this.loadTableFromR2(query.table)) ?? undefined;
        if (meta) {
          const count = meta.columns[0]?.pages.reduce((s, p) => s + p.rowCount, 0) ?? meta.totalRows;
          return this.json({ count });
        }
      }
      // With filters: use aggregate path
      query.aggregates = [{ fn: "count", column: "*" }];
      const result = await this.executeQuery(query);
      const count = (result.rows[0]?.["count_*"] as number) ?? 0;
      return this.json({ count });
    } catch (err) {
      return this.json({ error: err instanceof Error ? err.message : String(err) }, 500);
    }
  }

  private async handleExists(request: Request): Promise<Response> {
    const body = await request.json() as Record<string, unknown>;
    if (!body.table || typeof body.table !== "string") return this.json({ error: "Missing 'table'" }, 400);
    const query = this.parseQuery(body);
    query.limit = 1;
    try {
      const result = await this.executeQuery(query);
      return this.json({ exists: result.rowCount > 0 });
    } catch (err) {
      return this.json({ error: err instanceof Error ? err.message : String(err) }, 500);
    }
  }

  private async handleFirst(request: Request): Promise<Response> {
    const body = await request.json() as Record<string, unknown>;
    if (!body.table || typeof body.table !== "string") return this.json({ error: "Missing 'table'" }, 400);
    const query = this.parseQuery(body);
    query.limit = 1;
    try {
      const result = await this.executeQuery(query);
      return this.json({ row: result.rows[0] ?? null });
    } catch (err) {
      return this.json({ error: err instanceof Error ? err.message : String(err) }, 500);
    }
  }

  private async handleExplain(request: Request): Promise<Response> {
    const body = await request.json() as Record<string, unknown>;
    if (!body.table || typeof body.table !== "string") return this.json({ error: "Missing 'table'" }, 400);
    const query = this.parseQuery(body);
    try {
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

      const coalesced = coalesceRanges(ranges, 64 * 1024);
      const estimatedBytes = ranges.reduce((s, r) => s + r.length, 0);

      // Check dataset for fragment count
      const dataset = this.datasetCache.get(query.table);
      const fragments = dataset ? dataset.fragmentMetas.size : 1;

      const result: ExplainResult = {
        table: query.table,
        format: meta.format ?? "lance",
        totalRows: meta.totalRows,
        columns: colDetails,
        pagesTotal,
        pagesSkipped,
        pagesScanned: pagesTotal - pagesSkipped,
        estimatedBytes,
        estimatedR2Reads: coalesced.length,
        fragments,
        filters: query.filters.map(f => ({
          column: f.column,
          op: f.op,
          pushable: f.op !== "in" && f.op !== "neq",
        })),
        metaCached,
      };
      return this.json(result);
    } catch (err) {
      return this.json({ error: err instanceof Error ? err.message : String(err) }, 500);
    }
  }

  /** Execute query using page-level R2 Range reads + WASM compute. Never downloads full files. */
  private async executeQuery(query: QueryDescriptor): Promise<QueryResult> {
    const t0 = Date.now();

    // Result cache check (skip for vector search)
    if (query.cacheTTL && !query.vectorSearch) {
      const cacheKey = this.queryKey(query);
      const cached = this.resultCache.get(cacheKey);
      if (cached) return { ...cached, durationMs: 0 };
    }

    // Multi-fragment dataset path
    const dataset = this.datasetCache.get(query.table);
    if (dataset) {
      const result = await this.executeMultiFragment(query, dataset, t0);
      if (query.cacheTTL && !query.vectorSearch) {
        this.resultCache.setWithTTL(this.queryKey(query), result, query.cacheTTL);
      }
      return result;
    }

    let meta: TableMeta | undefined = this.footerCache.get(query.table);
    if (!meta) {
      meta = (await this.loadTableFromR2(query.table)) ?? undefined;
      if (!meta) throw new Error(`Table "${query.table}" not found`);
    }

    // Lance whole-file path: if no page byte ranges, load entire file into WASM
    const hasPages = meta.columns.some(c => c.pages.length > 0);
    if (!hasPages && meta.format === "lance" && meta.r2Key) {
      const result = await this.executeLanceWholeFile(query, meta, t0);
      if (query.cacheTTL && !query.vectorSearch) {
        this.resultCache.setWithTTL(this.queryKey(query), result, query.cacheTTL);
      }
      return result;
    }

    const result = await this.scanPages(query, meta, t0);
    if (query.cacheTTL && !query.vectorSearch) {
      this.resultCache.setWithTTL(this.queryKey(query), result, query.cacheTTL);
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
      const obj = await this.env.DATA_BUCKET.get(meta.r2Key);
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

    // Parse Lance v2 column metadata from the protobuf footer
    // The metadata uses nested encoding descriptors — extract row count from field 3 of the page sub-message
    const fileBytes = new Uint8Array(fileData);
    const footerStart = fileBytes.length - 40;
    const colMetaStart = Number(new DataView(fileData, footerStart, 8).getBigUint64(0, true));
    const colMetaOffsetsStart = Number(new DataView(fileData, footerStart + 8, 8).getBigUint64(0, true));
    const globalBufOffsetsStart = Number(new DataView(fileData, footerStart + 16, 8).getBigUint64(0, true));
    const numCols = new DataView(fileData, footerStart + 28, 4).getUint32(0, true);

    // Read global buffer offsets
    const bufOffsets: number[] = [];
    for (let i = globalBufOffsetsStart; i < footerStart; i += 8) {
      bufOffsets.push(Number(new DataView(fileData, i, 8).getBigUint64(0, true)));
    }

    // Parse each column's metadata to get column type and row count
    // Lance v2 column metadata: field 1 = encoding, field 2 = page data
    // Page data: field 1 = buffer ref, field 2 = data type, field 3 = row count
    const colInfos: { name: string; dtype: string; rowCount: number; dataOffset: number; dataSize: number }[] = [];
    for (let col = 0; col < numCols; col++) {
      const metaOffset = Number(new DataView(fileData, colMetaOffsetsStart + col * 8, 8).getBigUint64(0, true));
      const metaEnd = col + 1 < numCols
        ? Number(new DataView(fileData, colMetaOffsetsStart + (col + 1) * 8, 8).getBigUint64(0, true))
        : colMetaOffsetsStart;

      // Parse column protobuf (Lance v2)
      let pageBytes: Uint8Array | null = null;
      let pos = metaOffset;
      while (pos < metaEnd) {
        const tag = fileBytes[pos++];
        const fn2 = tag >> 3, wt = tag & 7;
        if (wt === 2) {
          let len = 0, shift = 0;
          while (pos < metaEnd) { const b = fileBytes[pos++]; len |= (b & 0x7f) << shift; shift += 7; if (!(b & 0x80)) break; }
          if (fn2 === 2) pageBytes = fileBytes.subarray(pos, pos + len);
          pos += len;
        } else if (wt === 0) { while (pos < metaEnd && fileBytes[pos++] & 0x80); }
        else if (wt === 1) pos += 8;
        else if (wt === 5) pos += 4;
        else break;
      }

      // Parse page sub-message for row count and data type
      let rowCount = 0;
      let dtypeCode = 0;
      if (pageBytes) {
        let pp = 0;
        while (pp < pageBytes.length) {
          const tag = pageBytes[pp++];
          const fn3 = tag >> 3, wt = tag & 7;
          if (wt === 0) {
            let val = 0, shift = 0;
            while (pp < pageBytes.length) { const b = pageBytes[pp++]; val |= (b & 0x7f) << shift; shift += 7; if (!(b & 0x80)) break; }
            if (fn3 === 3) rowCount = val;
          } else if (wt === 2) {
            let len = 0, shift = 0;
            while (pp < pageBytes.length) { const b = pageBytes[pp++]; len |= (b & 0x7f) << shift; shift += 7; if (!(b & 0x80)) break; }
            if (fn3 === 2 && len === 1) dtypeCode = pageBytes[pp]; // encoding hints
            pp += len;
          } else if (wt === 1) pp += 8;
          else if (wt === 5) pp += 4;
          else break;
        }
      }

      // Fall back to manifest totalRows if protobuf didn't encode row count
      if (rowCount === 0 && meta.totalRows > 0) {
        rowCount = meta.totalRows;
      }

      // Column name/type from footer-level metadata or manifest schema
      const colMeta = meta.columns[col];
      let colName = colMeta?.name ?? `column_${col}`;
      let dtype = colMeta?.dtype ?? "int64";

      // Override from manifest schema if available (covers v2 format where column meta has encoding paths)
      const dataset = this.datasetCache.get(query.table);
      if (dataset?.manifest.schema.length) {
        const leafFields = dataset.manifest.schema.filter(f => f.parentId === -1 || f.parentId === 0);
        const schemaField = leafFields[col];
        if (schemaField) {
          colName = schemaField.name;
          dtype = logicalTypeToDataType(schemaField.logicalType);
        }
      }

      const bytesPerValue = (dtype === "int32" || dtype === "float32") ? 4 : 8;
      colInfos.push({ name: colName, dtype, rowCount, dataOffset: 0, dataSize: rowCount * bytesPerValue });
    }

    this.log("info", "lance_fragment_parsed", {
      numCols, colInfos: colInfos.map(c => ({ name: c.name, dtype: c.dtype, rows: c.rowCount })),
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

    // Assemble rows
    const colNames = [...decodedColumns.keys()];
    let numRows = 0;
    for (const v of decodedColumns.values()) if (v.length > numRows) numRows = v.length;
    let rows: Row[] = [];
    for (let i = 0; i < numRows; i++) {
      const row: Row = {};
      for (const name of colNames) {
        const vals = decodedColumns.get(name);
        row[name] = vals && i < vals.length ? vals[i] : null;
      }
      rows.push(row);
    }

    // Apply filters, sort, limit in JS
    if (query.filters.length > 0) {
      rows = rows.filter(row => {
        for (const f of query.filters) {
          const val = row[f.column];
          if (val == null) return false;
          switch (f.op) {
            case "eq": if (val !== f.value) return false; break;
            case "neq": if (val === f.value) return false; break;
            case "gt": if (!(val > f.value)) return false; break;
            case "gte": if (!(val >= f.value)) return false; break;
            case "lt": if (!(val < f.value)) return false; break;
            case "lte": if (!(val <= f.value)) return false; break;
          }
        }
        return true;
      });
    }
    if (query.sortColumn) {
      const dir = query.sortDirection === "desc" ? -1 : 1;
      const sc = query.sortColumn;
      rows.sort((a, b) => {
        const va = a[sc], vb = b[sc];
        if (va == null && vb == null) return 0;
        if (va == null) return dir;
        if (vb == null) return -dir;
        return va < vb ? -dir : va > vb ? dir : 0;
      });
    }
    if (query.limit && query.limit > 0) rows = rows.slice(0, query.limit);

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

    // Cache-before-fetch: check WASM buffer pool for each range
    const columnData = new Map<string, ArrayBuffer[]>();
    let cacheHits = 0;
    let cacheMisses = 0;
    const uncachedRanges: typeof ranges = [];

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
        uncachedRanges.push(r);
      }
    }

    // Bounded prefetch pipeline: fire all R2 reads eagerly (up to 8 concurrent),
    // each fetch resolves independently so later logic doesn't block on earlier slow reads.
    // This overlaps R2 I/O: while fetch N is downloading, fetch N+1..N+7 are already in-flight.
    const r2Start = Date.now();
    let bytesRead = 0;

    if (uncachedRanges.length > 0) {
      const coalesced = coalesceRanges(uncachedRanges, 64 * 1024);

      // Fire all fetches eagerly — fetchBounded gates concurrency to 8
      const fetched = await fetchBounded(
        coalesced.map(c => () =>
          withRetry(() =>
            withTimeout(
              (async () => {
                const obj = await this.env.DATA_BUCKET.get(meta.r2Key, { range: { offset: c.offset, length: c.length } });
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

          // Populate cache for next time
          const cacheKey = `${meta.r2Key}:${sub.offset}:${sub.length}`;
          this.wasmEngine.cacheSet(cacheKey, slice);
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
                const dictObj = await this.env.DATA_BUCKET.get(meta.r2Key, {
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

      // Assemble rows from decoded columns
      const colNames = cols.map(c => c.name);
      let numRows = 0;
      for (const v of decodedColumns.values()) if (v.length > numRows) numRows = v.length;
      let rows: Row[] = [];
      for (let i = 0; i < numRows; i++) {
        const row: Row = {};
        for (const name of colNames) {
          const vals = decodedColumns.get(name);
          row[name] = vals && i < vals.length ? vals[i] : null;
        }
        rows.push(row);
      }

      // Apply filters in JS (coerce bigint↔number for cross-type comparison)
      if (query.filters.length > 0) {
        rows = rows.filter(row => {
          for (const f of query.filters) {
            const val = row[f.column];
            if (val == null) return false;
            // Coerce bigint↔number: int64 columns decode as bigint, JSON filter values are numbers
            let cv = val, cf = f.value;
            if (typeof cv === "bigint" && typeof cf === "number") cf = BigInt(Math.trunc(cf));
            else if (typeof cv === "number" && typeof cf === "bigint") cv = BigInt(Math.trunc(cv));
            switch (f.op) {
              case "eq": if (cv !== cf) return false; break;
              case "neq": if (cv === cf) return false; break;
              case "gt": if (!(cv > (cf as number | bigint | string))) return false; break;
              case "gte": if (!(cv >= (cf as number | bigint | string))) return false; break;
              case "lt": if (!(cv < (cf as number | bigint | string))) return false; break;
              case "lte": if (!(cv <= (cf as number | bigint | string))) return false; break;
            }
          }
          return true;
        });
      }

      // Apply sort
      if (query.sortColumn) {
        const dir = query.sortDirection === "desc" ? -1 : 1;
        const sc = query.sortColumn;
        rows.sort((a, b) => {
          const va = a[sc], vb = b[sc];
          if (va == null && vb == null) return 0;
          if (va == null) return dir;
          if (vb == null) return -dir;
          return va < vb ? -dir : va > vb ? dir : 0;
        });
      }

      // Apply limit
      if (query.limit && query.limit > 0) rows = rows.slice(0, query.limit);

      const wasmExecMs = Date.now() - wasmStart;
      return {
        rows, rowCount: rows.length, columns: colNames,
        bytesRead, pagesSkipped, durationMs: Date.now() - t0,
        r2ReadMs, wasmExecMs, cacheHits, cacheMisses,
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
    };
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
      const indexObj = await this.env.DATA_BUCKET.get(indexPath);
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

  private handleListTables(): Response {
    const tables = [...this.footerCache.entries()].map(([name, entry]) => ({
      name, columns: entry.value.columns.map(c => c.name), totalRows: entry.value.totalRows,
      updatedAt: entry.value.updatedAt, accessCount: entry.accessCount,
    }));
    return this.json({ tables });
  }

  private handleGetMeta(request: Request): Response {
    const table = new URL(request.url).searchParams.get("table");
    if (!table) return this.json({ error: "Missing ?table= parameter" }, 400);
    const meta = this.footerCache.get(table);
    if (!meta) return this.json({ error: `Table "${table}" not found` }, 404);
    return this.json(meta);
  }

  private handleDiagnostics(): Response {
    const cache = this.wasmEngine.cacheStats();
    return this.json({
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
    });
  }

  private async readColumnMeta(r2Key: string, footer: Footer): Promise<ColumnMeta[]> {
    const len = Number(footer.columnMetaOffsetsStart) - Number(footer.columnMetaStart);
    if (len <= 0) return [];
    const obj = await this.env.DATA_BUCKET.get(r2Key, { range: { offset: Number(footer.columnMetaStart), length: len } });
    if (!obj) return [];
    return parseColumnMetaFromProtobuf(await obj.arrayBuffer(), footer.numColumns);
  }

  private async loadTableFromR2(tableName: string): Promise<TableMeta | null> {
    const candidates = [
      `${tableName}.lance`, `${tableName}.parquet`, tableName,
      `data/${tableName}.lance`, `data/${tableName}.parquet`, `data/${tableName}`,
    ];
    for (const r2Key of candidates) {
      const head = await this.env.DATA_BUCKET.head(r2Key);
      if (!head) continue;

      const fileSize = BigInt(head.size);
      const tailSize = Math.min(Number(fileSize), 40);
      const obj = await this.env.DATA_BUCKET.get(r2Key, { range: { offset: Number(fileSize) - tailSize, length: tailSize } });
      if (!obj) continue;

      const tailBuf = await obj.arrayBuffer();
      const fmt = detectFormat(tailBuf);

      if (fmt === "parquet") {
        const footerLen = getParquetFooterLength(tailBuf);
        if (!footerLen) continue;

        const footerOffset = Number(fileSize) - footerLen - 8;
        const footerObj = await this.env.DATA_BUCKET.get(r2Key, { range: { offset: footerOffset, length: footerLen } });
        if (!footerObj) continue;

        const parquetMeta = parseParquetFooter(await footerObj.arrayBuffer());
        if (!parquetMeta) continue;

        const meta = parquetMetaToTableMeta(parquetMeta, r2Key, fileSize);
        meta.name = tableName;
        this.footerCache.set(tableName, meta);
        await this.state.storage.put(`table:${tableName}`, meta);
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
        await this.state.storage.put(`table:${tableName}`, meta);
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
      const listed = await this.env.DATA_BUCKET.list({ prefix: `${prefix}_versions/`, limit: 100 });
      const manifestKeys = listed.objects
        .filter(o => o.key.endsWith(".manifest"))
        .sort((a, b) => a.key.localeCompare(b.key));
      if (manifestKeys.length === 0) continue;

      // Read latest manifest
      const latestKey = manifestKeys[manifestKeys.length - 1].key;
      const manifestObj = await this.env.DATA_BUCKET.get(latestKey);
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
          head = await this.env.DATA_BUCKET.head(candidate);
          if (head) { fragKey = candidate; break; }
        }
        if (!head) continue;

        const fileSize = BigInt(head.size);
        const footerObj = await this.env.DATA_BUCKET.get(fragKey, { range: { offset: Number(fileSize) - 40, length: 40 } });
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
      this.evictDatasetCache();
      return dataset;
    }
    return null;
  }

  /** Register an Iceberg table by explicit metadata path (bypasses R2 list()). */
  private async handleRegisterIceberg(request: Request): Promise<Response> {
    const { table, metadataKey } = (await request.json()) as { table: string; metadataKey: string };
    if (!table || !metadataKey) return this.json({ error: "Missing table or metadataKey" }, 400);

    const result = await this.loadIcebergByKey(table, metadataKey);
    if (!result) return this.json({ error: "Failed to load Iceberg metadata" }, 500);
    return this.json({ registered: true, table, totalRows: result.totalRows, files: result.parquetFiles.length });
  }

  /** Load an Iceberg table from an explicit metadata.json key (no R2 list() needed). */
  private async loadIcebergByKey(tableName: string, metadataKey: string): Promise<IcebergDatasetMeta | null> {
    const metaObj = await this.env.DATA_BUCKET.get(metadataKey);
    if (!metaObj) return null;

    const metaJson = await metaObj.text();
    const icebergMeta = parseIcebergMetadata(metaJson);
    if (!icebergMeta) return null;

    // Derive prefix from metadataKey (e.g., "bench_iceberg_100k/metadata/v1.metadata.json" → "bench_iceberg_100k/")
    const prefix = metadataKey.replace(/metadata\/.*$/, "");

    const manifestListKey = `${prefix}${icebergMeta.manifestListPath}`;
    const manifestListObj = await this.env.DATA_BUCKET.get(manifestListKey);
    if (!manifestListObj) return null;

    const manifestListBytes = await manifestListObj.arrayBuffer();
    const parquetPaths = extractParquetPathsFromManifest(manifestListBytes);
    if (parquetPaths.length === 0) return null;

    // Load each Parquet file's metadata
    const fragmentMetas = new Map<number, TableMeta>();
    let totalRows = 0;
    for (let i = 0; i < parquetPaths.length; i++) {
      const parquetKey = parquetPaths[i].startsWith(prefix) ? parquetPaths[i] : `${prefix}${parquetPaths[i]}`;
      const head = await this.env.DATA_BUCKET.head(parquetKey);
      if (!head) continue;

      const fileSize = BigInt(head.size);
      const tailObj = await this.env.DATA_BUCKET.get(parquetKey, {
        range: { offset: Math.max(0, Number(fileSize) - 8), length: Math.min(8, Number(fileSize)) },
      });
      if (!tailObj) continue;

      const tailBuf = await tailObj.arrayBuffer();
      const footerLen = getParquetFooterLength(tailBuf);
      if (!footerLen) continue;

      const footerObj = await this.env.DATA_BUCKET.get(parquetKey, {
        range: { offset: Number(fileSize) - footerLen - 8, length: footerLen },
      });
      if (!footerObj) continue;

      const parquetFileMeta = parseParquetFooter(await footerObj.arrayBuffer());
      if (!parquetFileMeta) continue;

      const meta = parquetMetaToTableMeta(parquetFileMeta, parquetKey, fileSize);
      meta.name = parquetPaths[i];
      fragmentMetas.set(i, meta);
      totalRows += meta.totalRows;
    }

    if (fragmentMetas.size === 0) return null;

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
    this.evictDatasetCache();
    return dataset;
  }

  /** Discover an Iceberg table in R2 by listing metadata/ for .metadata.json files. */
  private async loadIcebergFromR2(tableName: string): Promise<IcebergDatasetMeta | null> {
    for (const prefix of [`${tableName}/`, `data/${tableName}/`]) {
      const listed = await this.env.DATA_BUCKET.list({ prefix: `${prefix}metadata/`, limit: 100 });
      const metadataKeys = listed.objects
        .filter(o => o.key.endsWith(".metadata.json"))
        .sort((a, b) => a.key.localeCompare(b.key));
      if (metadataKeys.length === 0) continue;

      // Read latest metadata JSON
      const latestKey = metadataKeys[metadataKeys.length - 1].key;
      const metaObj = await this.env.DATA_BUCKET.get(latestKey);
      if (!metaObj) continue;

      const metaJson = await metaObj.text();
      const icebergMeta = parseIcebergMetadata(metaJson);
      if (!icebergMeta) continue;

      // Read manifest list to get data file paths
      const manifestListKey = `${prefix}${icebergMeta.manifestListPath}`;
      const manifestListObj = await this.env.DATA_BUCKET.get(manifestListKey);
      if (!manifestListObj) continue;

      const manifestListBytes = await manifestListObj.arrayBuffer();
      const parquetPaths = extractParquetPathsFromManifest(manifestListBytes);
      if (parquetPaths.length === 0) continue;

      // Load each Parquet file's metadata
      const fragmentMetas = new Map<number, TableMeta>();
      let totalRows = 0;
      for (let i = 0; i < parquetPaths.length; i++) {
        const parquetKey = parquetPaths[i].startsWith(prefix) ? parquetPaths[i] : `${prefix}${parquetPaths[i]}`;
        const head = await this.env.DATA_BUCKET.head(parquetKey);
        if (!head) continue;

        const fileSize = BigInt(head.size);
        const tailObj = await this.env.DATA_BUCKET.get(parquetKey, {
          range: { offset: Math.max(0, Number(fileSize) - 8), length: Math.min(8, Number(fileSize)) },
        });
        if (!tailObj) continue;

        const tailBuf = await tailObj.arrayBuffer();
        const footerLen = getParquetFooterLength(tailBuf);
        if (!footerLen) continue;

        const footerObj = await this.env.DATA_BUCKET.get(parquetKey, {
          range: { offset: Number(fileSize) - footerLen - 8, length: footerLen },
        });
        if (!footerObj) continue;

        const parquetFileMeta = parseParquetFooter(await footerObj.arrayBuffer());
        if (!parquetFileMeta) continue;

        const meta = parquetMetaToTableMeta(parquetFileMeta, parquetKey, fileSize);
        meta.name = parquetPaths[i];
        fragmentMetas.set(i, meta);
        totalRows += meta.totalRows;
      }

      if (fragmentMetas.size === 0) continue;

      const dataset: IcebergDatasetMeta = {
        name: tableName, r2Prefix: prefix,
        schema: icebergMeta.schema, snapshotId: icebergMeta.currentSnapshotId,
        parquetFiles: parquetPaths, fragmentMetas, totalRows, updatedAt: Date.now(),
      };
      // Store in datasetCache for multi-fragment query execution
      this.datasetCache.set(tableName, {
        name: tableName, r2Prefix: prefix,
        manifest: { version: 0, fragments: parquetPaths.map((p, idx) => ({ id: idx, filePath: p, physicalRows: 0 })), totalRows, schema: [] },
        fragmentMetas, totalRows, updatedAt: Date.now(),
      });
      this.evictDatasetCache();
      return dataset;
    }
    return null;
  }

  /** Execute a query across multiple fragments of a dataset. */
  private async executeMultiFragment(query: QueryDescriptor, dataset: DatasetMeta, t0: number): Promise<QueryResult> {
    const fragments = [...dataset.fragmentMetas.values()];
    // Large datasets (>4 fragments): fan out to Fragment DOs
    if (fragments.length > 4) {
      return this.executeWithFragmentDOs(query, dataset, t0);
    }

    // Small datasets: scan locally
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
  private claimSlots(needed: number): number[] | null {
    const slots: number[] = [];
    for (let i = 0; i < FRAGMENT_POOL_MAX && slots.length < needed; i++) {
      if (!this.activeFragmentSlots.has(i)) slots.push(i);
    }
    if (slots.length < needed) return null; // not enough capacity
    for (const s of slots) this.activeFragmentSlots.add(s);
    return slots;
  }

  private releaseSlots(slots: number[]): void {
    for (const s of slots) this.activeFragmentSlots.delete(s);
  }

  /** Fan-out query to pooled Fragment DOs for parallel scanning.
   *  Uses Promise.allSettled for partial failure tolerance.
   *  Pool is per-datacenter (frag-{region}-slot-{N}), max FRAGMENT_POOL_MAX slots. */
  private async executeWithFragmentDOs(query: QueryDescriptor, dataset: DatasetMeta, t0: number): Promise<QueryResult> {
    const fragments = [...dataset.fragmentMetas.entries()];
    const slotsNeeded = Math.min(fragments.length, FRAGMENT_POOL_MAX - this.activeFragmentSlots.size);

    if (slotsNeeded <= 0) {
      // All slots busy — scan locally rather than queueing
      const partials: QueryResult[] = [];
      for (const [, meta] of fragments) {
        partials.push(await this.scanPages(query, meta, t0));
      }
      return mergeQueryResults(partials, query);
    }

    const slots = this.claimSlots(slotsNeeded)!;
    const chunkSize = Math.ceil(fragments.length / slots.length);

    // Partition fragments across claimed slots
    const groups: { r2Key: string; meta: TableMeta }[][] = [];
    for (let i = 0; i < fragments.length; i += chunkSize) {
      groups.push(fragments.slice(i, i + chunkSize).map(([, meta]) => ({
        r2Key: meta.r2Key, meta,
      })));
    }

    const region = (await this.state.storage.get<string>("region")) ?? "default";

    try {
      const settled = await Promise.allSettled(groups.map(async (group, idx) => {
        const doName = `frag-${region}-slot-${slots[idx]}`;
        const doId = this.env.FRAGMENT_DO.idFromName(doName);
        const fragmentDo = this.env.FRAGMENT_DO.get(doId);

        const resp = await withTimeout(
          fragmentDo.fetch(new Request("http://internal/scan", {
            method: "POST",
            body: JSON.stringify({ fragments: group, query }, bigIntReplacer),
            headers: { "content-type": "application/json" },
          })),
          FRAGMENT_TIMEOUT_MS,
        );

        return resp.json() as Promise<QueryResult>;
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

  /** Stream query results as NDJSON. */
  private async handleQueryStream(request: Request): Promise<Response> {
    const query = (await request.json()) as QueryDescriptor;
    const result = await this.executeQuery(query);

    const { readable, writable } = new TransformStream<Uint8Array>();
    const writer = writable.getWriter();
    const encoder = new TextEncoder();

    (async () => {
      try {
        for (const row of result.rows) {
          await writer.write(encoder.encode(JSON.stringify(row, bigIntReplacer) + "\n"));
        }
      } finally {
        await writer.close();
      }
    })();

    return new Response(readable, {
      headers: { "content-type": "application/x-ndjson" },
    });
  }

}
