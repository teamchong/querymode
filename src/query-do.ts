import type { ColumnMeta, Env, Footer, Row, TableMeta, DatasetMeta, IcebergDatasetMeta, QueryResult } from "./types.js";
import type { QueryDescriptor } from "./client.js";
import { parseFooter, parseColumnMetaFromProtobuf } from "./footer.js";
import { parseManifest } from "./manifest.js";
import { detectFormat, getParquetFooterLength, parseParquetFooter, parquetMetaToTableMeta } from "./parquet.js";
import { parseIcebergMetadata, extractParquetPathsFromManifest } from "./iceberg.js";
import { canSkipPage, bigIntReplacer, assembleRows } from "./decode.js";
import { instantiateWasm, type WasmEngine } from "./wasm-engine.js";
import { mergeQueryResults } from "./merge.js";
import { coalesceRanges, fetchBounded, withRetry, withTimeout } from "./coalesce.js";

const FRAGMENT_POOL_MAX = 20; // Max Fragment DO slots per datacenter (idle slots cost nothing)
const R2_TIMEOUT_MS = 10_000;
const FRAGMENT_TIMEOUT_MS = 25_000;

/**
 * Query DO — per-region reader with cached footers.
 * WASM (Zig SIMD) handles all compute; TS handles I/O orchestration only.
 */
export class QueryDO implements DurableObject {
  private state: DurableObjectState;
  private env: Env;
  private footerCache = new Map<string, TableMeta>();
  private datasetCache = new Map<string, DatasetMeta>();
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
      case "/tables":       return this.handleListTables();
      case "/meta":         return this.handleGetMeta(request);
      case "/diagnostics":  return this.handleDiagnostics();
      default:              return new Response("Not found", { status: 404 });
    }
  }

  private async ensureInitialized(): Promise<void> {
    if (this.initialized) return;
    this.initialized = true;

    const stored = await this.state.storage.list<TableMeta>({ prefix: "table:" });
    for (const [key, meta] of stored) this.footerCache.set(key.replace("table:", ""), meta);

    this.wasmEngine = await instantiateWasm(this.env.QUERYMODE_WASM);

    // Register with Master for invalidation broadcasts
    this.registerWithMaster();
  }

  private registerWithMaster(): void {
    const master = this.env.MASTER_DO.get(this.env.MASTER_DO.idFromName("master"));
    this.state.storage.get<string>("region").then(region => {
      master.fetch(new Request("http://internal/register", {
        method: "POST",
        body: JSON.stringify({ queryDoId: this.state.id.toString(), region: region ?? "unknown" }),
        headers: { "content-type": "application/json" },
      })).then(async (resp) => {
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
      }).catch(() => {
        this.log("warn", "master_register_failed", { note: "will retry on next request" });
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
    };

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

    const meta: TableMeta = {
      name: body.table, footer, format: fmt, columns,
      totalRows: columns[0]?.pages.reduce((s, p) => s + p.rowCount, 0) ?? 0,
      fileSize: body.fileSize ? BigInt(body.fileSize) : 0n,
      r2Key: body.r2Key, updatedAt: body.timestamp,
    };

    this.footerCache.set(body.table, meta);
    this.wasmEngine.clearTable(body.table);
    this.wasmEngine.cacheClear();
    await this.state.storage.put(`table:${body.table}`, meta);
    return this.json({ updated: true, table: body.table });
  }

  private async handleQuery(request: Request): Promise<Response> {
    const requestId = request.headers.get("x-querymode-request-id") ?? crypto.randomUUID();
    const query = (await request.json()) as QueryDescriptor;
    const result = await this.executeQuery(query);
    result.requestId = requestId;
    this.log("info", "query_complete", {
      requestId, table: query.table, rowCount: result.rowCount,
      bytesRead: result.bytesRead, durationMs: result.durationMs,
      r2ReadMs: result.r2ReadMs, wasmExecMs: result.wasmExecMs,
      cacheHits: result.cacheHits, cacheMisses: result.cacheMisses,
    });
    return this.json(result);
  }

  /** Execute query using page-level R2 Range reads + WASM compute. Never downloads full files. */
  private async executeQuery(query: QueryDescriptor): Promise<QueryResult> {
    const t0 = Date.now();

    // Multi-fragment dataset path
    const dataset = this.datasetCache.get(query.table);
    if (dataset) return this.executeMultiFragment(query, dataset, t0);

    let meta: TableMeta | undefined = this.footerCache.get(query.table);
    if (!meta) {
      meta = (await this.loadTableFromR2(query.table)) ?? undefined;
      if (!meta) throw new Error(`Table "${query.table}" not found`);
    }

    return this.scanPages(query, meta, t0);
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

    // Build per-page ranges, applying page-level skip
    const ranges: { column: string; offset: number; length: number }[] = [];
    let pagesSkipped = 0;
    for (const col of cols) {
      for (const page of col.pages) {
        if (!query.vectorSearch && canSkipPage(page, query.filters, col.name)) { pagesSkipped++; continue; }
        ranges.push({ column: col.name, offset: Number(page.byteOffset), length: page.byteLength });
      }
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

    // Fetch only uncached ranges from R2 with retry + timeout
    const r2Start = Date.now();
    let bytesRead = 0;

    if (uncachedRanges.length > 0) {
      const coalesced = coalesceRanges(uncachedRanges, 64 * 1024);
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

    // Zero-copy WASM path: register raw page data per-column, execute SQL in WASM
    const wasmStart = Date.now();
    this.wasmEngine.exports.resetHeap();
    for (const col of cols) {
      const pages = columnData.get(col.name);
      if (!pages?.length) continue;
      if (!this.wasmEngine.registerColumn(query.table, col.name, col.dtype, pages, col.pages, col.listDimension)) {
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
    const tables = [...this.footerCache.entries()].map(([name, meta]) => ({
      name, columns: meta.columns.map(c => c.name), totalRows: meta.totalRows, updatedAt: meta.updatedAt,
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
      tables: [...this.footerCache.entries()].map(([name, m]) => ({
        name, totalRows: m.totalRows, columns: m.columns.length, updatedAt: m.updatedAt,
      })),
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
        const fragKey = `${prefix}${frag.filePath}`;
        const head = await this.env.DATA_BUCKET.head(fragKey);
        if (!head) continue;

        const fileSize = BigInt(head.size);
        const footerObj = await this.env.DATA_BUCKET.get(fragKey, { range: { offset: Number(fileSize) - 40, length: 40 } });
        if (!footerObj) continue;

        const footer = parseFooter(await footerObj.arrayBuffer());
        if (!footer) continue;

        const columns = await this.readColumnMeta(fragKey, footer);
        fragmentMetas.set(frag.id, {
          name: frag.filePath, footer, columns,
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
      return dataset;
    }
    return null;
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
        manifest: { version: 0, fragments: parquetPaths.map((p, idx) => ({ id: idx, filePath: p, physicalRows: 0 })), totalRows },
        fragmentMetas, totalRows, updatedAt: Date.now(),
      });
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
      partials.push(await this.scanPages(query, meta, t0));
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
