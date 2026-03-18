/**
 * LocalExecutor — query execution on local filesystem (Node/Bun).
 *
 * Extracted from index.ts so that local.ts can import it without
 * triggering Cloudflare DO module loads (wasm-module.ts uses Wrangler
 * .wasm import syntax that crashes in Node).
 */
import type { QueryDescriptor, QueryExecutor } from "./client.js";
import type { AppendResult, ColumnMeta, DataType, DiffResult, ExplainResult, PageInfo, QueryResult, Row, TableMeta, DatasetMeta, VersionInfo } from "./types.js";
import { queryReferencedColumns, NULL_SENTINEL } from "./types.js";
import { parseFooter, parseColumnMetaFromProtobuf, FOOTER_SIZE } from "./footer.js";
import { parseManifest } from "./manifest.js";
import { detectFormat, getParquetFooterLength, parseParquetFooter, parquetMetaToTableMeta } from "./parquet.js";
import { assembleRows, canSkipFragment } from "./decode.js";
import { coalesceRanges, autoCoalesceGap } from "./coalesce.js";
import { instantiateWasm, type WasmEngine } from "./wasm-engine.js";

const textEncoder = new TextEncoder();
import { VipCache } from "./vip-cache.js";
import { QueryModeError } from "./errors.js";
import { parseLanceV2Columns, lanceV2ToColumnMeta, computeLanceV2Stats } from "./lance-v2.js";
import { buildPipeline, drainPipeline, DEFAULT_MEMORY_BUDGET, canSkipPageMultiCol, type FragmentSource, type PipelineOptions } from "./operators.js";

/**
 * Executor for local mode (Node/Bun).
 * Reads Lance/Parquet files directly from the filesystem or via HTTP.
 * Footer is parsed on first access and cached in-process.
 */
export interface LocalExecutorOptions {
  wasmModule?: WebAssembly.Module;
  /** Memory budget in bytes for external sort (default 256MB). */
  memoryBudgetBytes?: number;
}

export class LocalExecutor implements QueryExecutor {
  private metaCache: Map<string, { columns: ColumnMeta[]; fileSize: number }> = new Map();
  private datasetCache: Map<string, DatasetMeta> = new Map();
  private resultCache = new VipCache<string, QueryResult>(200, 2);
  private wasmModule?: WebAssembly.Module;
  private wasmEngine?: WasmEngine;
  private memoryBudgetBytes?: number;
  /** Pluggable format readers (CSV, JSON, Arrow IPC, etc.) — lazy-initialized. */
  private readerRegistry?: import("./reader.js").ReaderRegistry;
  /** Cache of reader-produced FragmentSources keyed by table path. */
  private readerFragmentCache: Map<string, import("./operators.js").FragmentSource[]> = new Map();

  constructor(wasmModuleOrOpts?: WebAssembly.Module | LocalExecutorOptions) {
    if (wasmModuleOrOpts && typeof wasmModuleOrOpts === "object" && "wasmModule" in wasmModuleOrOpts) {
      this.wasmModule = wasmModuleOrOpts.wasmModule;
      this.memoryBudgetBytes = wasmModuleOrOpts.memoryBudgetBytes;
    } else {
      this.wasmModule = wasmModuleOrOpts as WebAssembly.Module | undefined;
    }
  }

  /** Get or create the reader registry with built-in format readers. */
  private async getReaderRegistry(): Promise<import("./reader.js").ReaderRegistry> {
    if (this.readerRegistry) return this.readerRegistry;
    const { ReaderRegistry, FileDataSource } = await import("./reader.js");
    const registry = new ReaderRegistry();
    // Register built-in readers
    try {
      const { CsvReader } = await import("./readers/csv-reader.js");
      registry.register(new CsvReader());
    } catch { /* csv reader not available */ }
    try {
      const { JsonReader } = await import("./readers/json-reader.js");
      registry.register(new JsonReader());
    } catch { /* json reader not available */ }
    try {
      const { ArrowReader } = await import("./readers/arrow-reader.js");
      registry.register(new ArrowReader());
    } catch { /* arrow reader not available */ }
    this.readerRegistry = registry;
    return registry;
  }

  private async getWasm(): Promise<WasmEngine> {
    if (this.wasmEngine) return this.wasmEngine;
    if (!this.wasmModule) {
      // Load WASM from file system (Node/Bun)
      const fs = await import("node:fs/promises");
      const path = await import("node:path");
      const wasmPath = path.join(import.meta.dirname ?? ".", "wasm", "querymode.wasm");
      const wasmBytes = await fs.readFile(wasmPath);
      // WebAssembly.compile not in Cloudflare types but available in Node/Bun
      this.wasmModule = await (WebAssembly as unknown as { compile(b: BufferSource): Promise<WebAssembly.Module> }).compile(wasmBytes);
    }
    this.wasmEngine = await instantiateWasm(this.wasmModule!);
    return this.wasmEngine;
  }

  /** Append rows to a local Lance dataset. Builds fragment via WASM and writes to disk. */
  async append(tablePath: string, rows: Record<string, unknown>[]): Promise<AppendResult> {
    const fs = await import("node:fs/promises");
    const pathMod = await import("node:path");
    const wasm = await this.getWasm();

    // Ensure target is a directory (not a file like .parquet)
    const stat = await fs.stat(tablePath).catch(() => null);
    if (stat && !stat.isDirectory()) {
      throw new QueryModeError("INVALID_FORMAT", `append() only works on Lance dataset directories, not files: ${tablePath}`);
    }

    // Ensure dataset directory structure exists
    const dataDir = pathMod.join(tablePath, "data");
    const versionsDir = pathMod.join(tablePath, "_versions");
    await fs.mkdir(dataDir, { recursive: true });
    await fs.mkdir(versionsDir, { recursive: true });

    // Convert rows to column arrays
    const columnNames = Object.keys(rows[0]);
    const columnArrays: { name: string; dtype: string; values: ArrayBufferLike }[] = [];

    for (const colName of columnNames) {
      const sample = rows.find(r => r[colName] != null)?.[colName];
      if (sample === undefined) continue;

      if (typeof sample === "number") {
        if (Number.isInteger(sample)) {
          const arr = new BigInt64Array(rows.length);
          for (let i = 0; i < rows.length; i++) { const v = rows[i][colName]; arr[i] = typeof v === "bigint" ? v : BigInt(Math.trunc(Number(v ?? 0))); }
          columnArrays.push({ name: colName, dtype: "int64", values: arr.buffer });
        } else {
          const arr = new Float64Array(rows.length);
          for (let i = 0; i < rows.length; i++) arr[i] = rows[i][colName] as number;
          columnArrays.push({ name: colName, dtype: "float64", values: arr.buffer });
        }
      } else if (typeof sample === "bigint") {
        const arr = new BigInt64Array(rows.length);
        for (let i = 0; i < rows.length; i++) arr[i] = rows[i][colName] as bigint;
        columnArrays.push({ name: colName, dtype: "int64", values: arr.buffer });
      } else if (typeof sample === "string") {
        const enc = textEncoder;
        const parts: Uint8Array[] = [];
        let totalLen = 0;
        for (const row of rows) {
          const str = enc.encode(String(row[colName] ?? ""));
          const header = new Uint8Array(4);
          new DataView(header.buffer).setUint32(0, str.length, true);
          parts.push(header, str);
          totalLen += 4 + str.length;
        }
        const buf = new Uint8Array(totalLen);
        let off = 0;
        for (const p of parts) { buf.set(p, off); off += p.length; }
        columnArrays.push({ name: colName, dtype: "utf8", values: buf.buffer });
      }
    }

    // Build fragment via WASM
    const fragmentBytes = wasm.buildFragment(columnArrays);

    // Write data file
    const uuid = crypto.randomUUID();
    const dataFilePath = `data/${uuid}.lance`;
    const fullDataPath = pathMod.join(tablePath, dataFilePath);
    await fs.writeFile(fullDataPath, fragmentBytes);

    // Read current version
    const latestPath = pathMod.join(versionsDir, "_latest");
    let currentVersion = 0;
    try {
      const text = await fs.readFile(latestPath, "utf8");
      currentVersion = parseInt(text.trim(), 10) || 0;
    } catch { /* file doesn't exist yet */ }

    const newVersion = currentVersion + 1;

    // Write _latest
    await fs.writeFile(latestPath, String(newVersion));

    // Invalidate meta + result caches
    this.metaCache.delete(tablePath);
    this.datasetCache.delete(tablePath);
    this.resultCache.invalidateByPrefix(`qr:${tablePath}:`);

    return {
      version: newVersion,
      dataFilePath,
      retries: 0,
      rowsWritten: rows.length,
    };
  }

  /** Count matching rows. No-filter case uses metadata only (zero I/O). */
  async count(query: QueryDescriptor): Promise<number> {
    const meta = await this.getOrLoadMeta(query.table);
    if (query.filters.length === 0 && !query.filterGroups?.length && !query.aggregates?.length && !query.groupBy?.length && !query.distinct && !query.join && !query.vectorSearch && !query.setOperation && !query.subqueryIn && !query.computedColumns?.length) {
      return meta.columns[0]?.pages.reduce((s, p) => s + p.rowCount, 0) ?? 0;
    }
    // With filters: fall through to aggregate path
    const desc = { ...query, aggregates: [{ fn: "count" as const, column: "*" }] };
    const result = await this.execute(desc);
    return (result.rows[0]?.["count_*"] as number) ?? 0;
  }

  async exists(query: QueryDescriptor): Promise<boolean> {
    const desc = { ...query, limit: 1 };
    const result = await this.execute(desc);
    return result.rowCount > 0;
  }

  async first(query: QueryDescriptor): Promise<Row | null> {
    const desc = { ...query, limit: 1 };
    const result = await this.execute(desc);
    return result.rows[0] ?? null;
  }

  async explain(query: QueryDescriptor): Promise<ExplainResult> {
    const meta = await this.getOrLoadMeta(query.table);
    const { columns } = meta;
    const neededCols = queryReferencedColumns(query, columns.map(c => c.name));
    const projectedColumns = columns.filter(c => neededCols.has(c.name));

    let pagesTotal = 0;
    let pagesSkipped = 0;
    let estimatedRows = 0;
    const ranges: { column: string; offset: number; length: number }[] = [];
    const colDetails: ExplainResult["columns"] = [];

    // Uniform page-level skip across all columns to match actual query behavior
    const maxPages = projectedColumns.reduce((m, c) => Math.max(m, c.pages.length), 0);
    const keptPages = new Set<number>();
    for (let pi = 0; pi < maxPages; pi++) {
      if (!query.vectorSearch && canSkipPageMultiCol(projectedColumns, pi, query.filters, query.filterGroups)) {
        pagesSkipped += projectedColumns.length;
      } else {
        keptPages.add(pi);
      }
    }
    pagesTotal = maxPages * projectedColumns.length;

    const firstCol = projectedColumns[0];
    for (const col of projectedColumns) {
      let colBytes = 0;
      let colPages = 0;
      for (let pi = 0; pi < col.pages.length; pi++) {
        if (!keptPages.has(pi)) continue;
        const page = col.pages[pi];
        colPages++;
        colBytes += page.byteLength;
        ranges.push({ column: col.name, offset: Number(page.byteOffset), length: page.byteLength });
        if (col === firstCol) estimatedRows += page.rowCount;
      }
      colDetails.push({ name: col.name, dtype: col.dtype as DataType, pages: colPages, bytes: colBytes });
    }

    const coalesced = coalesceRanges(ranges, autoCoalesceGap(ranges));
    const estimatedBytes = ranges.reduce((s, r) => s + r.length, 0);
    const totalRows = columns[0]?.pages.reduce((s, p) => s + p.rowCount, 0) ?? 0;

    // Detect format from file
    const format = await this.detectFileFormat(query.table);

    // Fragment-level pruning stats
    const dataset = this.datasetCache.get(query.table);
    const totalFragments = dataset ? dataset.fragmentMetas.size : 1;
    let fragmentsSkipped = 0;
    if (dataset) {
      for (const fragMeta of dataset.fragmentMetas.values()) {
        if (canSkipFragment(fragMeta, query.filters, query.filterGroups)) fragmentsSkipped++;
      }
    }

    return {
      table: query.table,
      format,
      totalRows,
      columns: colDetails,
      pagesTotal,
      pagesSkipped,
      pagesScanned: pagesTotal - pagesSkipped,
      estimatedBytes,
      estimatedR2Reads: coalesced.length,
      estimatedRows,
      fragments: totalFragments,
      fragmentsSkipped,
      fragmentsScanned: totalFragments - fragmentsSkipped,
      fanOut: false,
      filters: [
        ...query.filters.map(f => ({ column: f.column, op: f.op, pushable: true })),
        ...(query.filterGroups ?? []).flatMap(g => g.map(f => ({ column: f.column, op: f.op, pushable: true }))),
      ],
      metaCached: this.metaCache.has(query.table),
    };
  }

  /** Detect file format from tail bytes. */
  private async detectFileFormat(table: string): Promise<"lance" | "parquet" | "iceberg"> {
    const isUrl = table.startsWith("http://") || table.startsWith("https://");
    if (isUrl) {
      const resp = await fetch(table, { method: "HEAD" });
      const size = Number(resp.headers.get("content-length") ?? 0);
      const tailSize = Math.min(size, FOOTER_SIZE);
      const tailResp = await fetch(table, { headers: { Range: `bytes=${size - tailSize}-${size - 1}` } });
      const tailAb = await tailResp.arrayBuffer();
      return detectFormat(tailAb) ?? "lance";
    }
    const fs = await import("node:fs/promises");
    const stat = await fs.stat(table);
    if (stat.isDirectory()) return "lance"; // Lance dataset directories are always Lance format
    const fileSize = Number(stat.size);
    const tailSize = Math.min(fileSize, FOOTER_SIZE);
    const handle = await fs.open(table, "r");
    try {
      const tailBuf = Buffer.alloc(tailSize);
      await handle.read(tailBuf, 0, tailSize, fileSize - tailSize);
      const tailAb = tailBuf.buffer.slice(tailBuf.byteOffset, tailBuf.byteOffset + tailBuf.byteLength);
      return detectFormat(tailAb) ?? "lance";
    } finally {
      await handle.close();
    }
  }

  async *cursor(query: QueryDescriptor, batchSize: number): AsyncIterable<Row[]> {
    // Use the streaming pipeline — it handles files, URLs, and dataset directories
    const result = await this.execute(query);
    for (let i = 0; i < result.rows.length; i += batchSize) {
      yield result.rows.slice(i, i + batchSize);
    }
  }

  /** Create a FragmentSource for a single file (local or URL). */
  private async makeFragmentSource(
    table: string,
    projectedColumns: ColumnMeta[],
    isUrl: boolean,
  ): Promise<FragmentSource> {
    if (isUrl) {
      return {
        columns: projectedColumns,
        async readPage(_col: ColumnMeta, page: PageInfo): Promise<ArrayBuffer> {
          const start = Number(page.byteOffset);
          const end = start + page.byteLength - 1;
          const resp = await fetch(table, { headers: { Range: `bytes=${start}-${end}` } });
          return resp.arrayBuffer();
        },
      };
    }
    const fsMod = await import("node:fs/promises");
    return {
      columns: projectedColumns,
      async readPage(_col: ColumnMeta, page: PageInfo): Promise<ArrayBuffer> {
        const handle = await fsMod.open(table, "r");
        try {
          const buf = Buffer.alloc(page.byteLength);
          await handle.read(buf, 0, page.byteLength, Number(page.byteOffset));
          return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
        } finally {
          await handle.close();
        }
      },
    };
  }

  /** Get or load table metadata (footer + columns). Caches in-memory. Handles both files and dataset directories. */
  private async getOrLoadMeta(table: string): Promise<{ columns: ColumnMeta[]; fileSize: number }> {
    let cached = this.metaCache.get(table);
    if (cached) return cached;
    const isUrl = table.startsWith("http://") || table.startsWith("https://");
    if (!isUrl) {
      const fs = await import("node:fs/promises");
      const stat = await fs.stat(table).catch((err: unknown) => {
        throw QueryModeError.from(err, { table });
      });
      if (stat.isDirectory()) {
        // Lance dataset directory — load via executeDatasetQuery path to populate cache
        const dataset = await this.getOrLoadDataset(table);
        const firstMeta = dataset.fragmentMetas.values().next().value;
        const columns = firstMeta?.columns ?? [];
        const fileSize = Number(firstMeta?.fileSize ?? 0n);
        cached = { columns, fileSize };
        this.metaCache.set(table, cached);
        return cached;
      }
    }
    cached = isUrl ? await this.loadMetaFromUrl(table) : await this.loadMetaFromFile(table);
    this.metaCache.set(table, cached);
    if (this.metaCache.size > 1000) {
      const firstKey = this.metaCache.keys().next().value;
      if (firstKey) this.metaCache.delete(firstKey);
    }
    return cached;
  }

  /** Build a cache key from query descriptor — no serialization. */
  private queryCacheKey(query: QueryDescriptor): string {
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
    if (query.setOperation) { feed(query.setOperation.mode); feed("\0"); feed(this.queryCacheKey(query.setOperation.right)); feed("\0"); }
    if (query.subqueryIn) for (const sq of query.subqueryIn) { feed(sq.column); feed("\0"); for (const v of sq.valueSet) { feed(v); feed("\0"); } }
    if (query.join) { feed(query.join.type ?? "inner"); feed("\0"); feed(query.join.leftKey); feed("\0"); feed(query.join.rightKey); feed("\0"); feed(this.queryCacheKey(query.join.right)); feed("\0"); }
    return `qr:${query.table}:${(h >>> 0).toString(36)}`;
  }

  async execute(query: QueryDescriptor): Promise<QueryResult> {
    const startTime = Date.now();
    const isUrl = query.table.startsWith("http://") || query.table.startsWith("https://");

    // Check result cache (skip for vector search — non-deterministic with IVF-PQ)
    if (query.cacheTTL && !query.vectorSearch) {
      const cacheKey = this.queryCacheKey(query);
      const cached = this.resultCache.get(cacheKey);
      if (cached) return { ...cached, cacheHit: true, durationMs: Date.now() - startTime };
    }

    // Check if this is a Lance dataset directory (has _versions/ subdir)
    if (!isUrl) {
      const fs = await import("node:fs/promises");
      const stat = await fs.stat(query.table).catch((err: NodeJS.ErrnoException) => {
        if (err.code === 'ENOENT') throw QueryModeError.from(err, { table: query.table, operation: "execute" });
        throw err;
      });
      if (stat?.isDirectory()) {
        return this.executeDatasetQuery(query, startTime);
      }
    }

    // Step 1: Get or cache table metadata (footer + column meta)
    const metaStart = Date.now();
    let meta = this.metaCache.get(query.table);
    if (!meta) {
      meta = isUrl
        ? await this.loadMetaFromUrl(query.table)
        : await this.loadMetaFromFile(query.table);
      this.metaCache.set(query.table, meta);
      if (this.metaCache.size > 1000) {
        const firstKey = this.metaCache.keys().next().value;
        if (firstKey) this.metaCache.delete(firstKey);
      }
    }
    const metaMs = Date.now() - metaStart;

    const { columns } = meta;
    const columnNames = new Set(columns.map(c => c.name));

    // Validate column references
    for (const p of query.projections) {
      if (!columnNames.has(p)) {
        throw new QueryModeError("COLUMN_NOT_FOUND", `Column "${p}" not found in ${query.table}. Available: ${[...columnNames].join(", ")}`);
      }
    }
    for (const f of query.filters) {
      if (!columnNames.has(f.column)) {
        throw new QueryModeError("COLUMN_NOT_FOUND", `Filter column "${f.column}" not found in ${query.table}. Available: ${[...columnNames].join(", ")}`);
      }
    }
    if (query.sortColumn && !columnNames.has(query.sortColumn)) {
      const aggAliases = new Set((query.aggregates ?? []).map(a => a.alias ?? `${a.fn}_${a.column}`));
      const groupCols = new Set(query.groupBy ?? []);
      if (!aggAliases.has(query.sortColumn) && !groupCols.has(query.sortColumn)) {
        throw new QueryModeError("COLUMN_NOT_FOUND", `Sort column "${query.sortColumn}" not found in ${query.table}. Available: ${[...columnNames].join(", ")}`);
      }
    }

    // Step 2: Determine columns to fetch (projection + all referenced columns)
    const neededColumns = queryReferencedColumns(query, columns.map(c => c.name));
    const projectedColumns = columns.filter(c => neededColumns.has(c.name));

    // Vector search still uses the legacy all-at-once path (needs full embeddings for WASM SIMD)
    if (query.vectorSearch) {
      return this.executeVectorSearch(query, projectedColumns, isUrl, startTime);
    }

    // Step 3: Build streaming pipeline
    const wasm = await this.getWasm();
    // Use reader-produced fragments if available (CSV, JSON, Arrow IPC, etc.)
    const readerFragments = this.readerFragmentCache.get(query.table);
    const fragment = readerFragments ? readerFragments[0] : await this.makeFragmentSource(query.table, projectedColumns, isUrl);
    const { FsSpillBackend } = await import("./operators.js");
    const pipeOpts: PipelineOptions = {
      memoryBudgetBytes: this.memoryBudgetBytes ?? DEFAULT_MEMORY_BUDGET,
      spill: new FsSpillBackend(),
    };

    // If join is specified, build left + right pipelines and combine with HashJoinOperator
    if (query.join) {
      return this.executeJoin(query, [fragment], wasm, pipeOpts, startTime);
    }

    const outputColumns = query.projections.length > 0 ? query.projections : projectedColumns.map(c => c.name);
    const { pipeline, scan, wasmAgg } = buildPipeline([fragment], query, wasm, outputColumns, pipeOpts);

    // Step 4: Drain pipeline
    const rows = await drainPipeline(pipeline);

    const scanMs = wasmAgg?.scanMs ?? scan?.scanMs ?? 0;
    const pipelineMs = Date.now() - startTime - metaMs;
    const result: QueryResult & { scanMs?: number; pipelineMs?: number; metaMs?: number } = {
      rows,
      rowCount: rows.length,
      columns: outputColumns,
      bytesRead: wasmAgg?.bytesRead ?? scan?.bytesRead ?? 0,
      pagesSkipped: wasmAgg?.pagesSkipped ?? scan?.pagesSkipped ?? 0,
      durationMs: Date.now() - startTime,
      cacheHit: false,
      scanMs,
      pipelineMs,
      metaMs,
    };

    if (query.cacheTTL) {
      this.resultCache.setWithTTL(this.queryCacheKey(query), result, query.cacheTTL);
    }

    return result;
  }

  /** Legacy vector search path — needs all embeddings materialized for WASM SIMD. */
  private async executeVectorSearch(
    query: QueryDescriptor,
    projectedColumns: ColumnMeta[],
    isUrl: boolean,
    startTime: number,
  ): Promise<QueryResult> {
    const pageRanges: { column: string; offset: bigint; length: number }[] = [];
    let pagesSkipped = 0;

    // Uniform page-level skip: decide once per page index across all columns to avoid row misalignment.
    const maxPages = projectedColumns.reduce((m, c) => Math.max(m, c.pages.length), 0);
    const keptPageIndices: number[] = [];
    for (let pi = 0; pi < maxPages; pi++) {
      if (canSkipPageMultiCol(projectedColumns, pi, query.filters, query.filterGroups)) {
        pagesSkipped += projectedColumns.length;
        continue;
      }
      keptPageIndices.push(pi);
    }
    for (const col of projectedColumns) {
      for (const pi of keptPageIndices) {
        const page = col.pages[pi];
        if (!page) continue;
        pageRanges.push({ column: col.name, offset: page.byteOffset, length: page.byteLength });
      }
    }

    const columnData = new Map<string, ArrayBuffer[]>();
    let bytesRead = 0;

    if (isUrl) {
      const pageBuffers = await Promise.all(
        pageRanges.map(async (range) => {
          const start = Number(range.offset);
          const end = start + range.length - 1;
          const resp = await fetch(query.table, {
            headers: { Range: `bytes=${start}-${end}` },
          });
          if (!resp.ok && resp.status !== 206) return null;
          return { column: range.column, data: await resp.arrayBuffer() };
        }),
      );
      for (const buf of pageBuffers) {
        if (!buf) continue;
        bytesRead += buf.data.byteLength;
        const existing = columnData.get(buf.column) ?? [];
        existing.push(buf.data);
        columnData.set(buf.column, existing);
      }
    } else {
      const fs = await import("node:fs/promises");
      const handle = await fs.open(query.table, "r");
      try {
        for (const range of pageRanges) {
          const buf = Buffer.alloc(range.length);
          await handle.read(buf, 0, range.length, Number(range.offset));
          bytesRead += range.length;
          const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
          const existing = columnData.get(range.column) ?? [];
          existing.push(ab);
          columnData.set(range.column, existing);
        }
      } finally {
        await handle.close();
      }
    }

    const wasm = await this.getWasm();
    const rows = assembleRows(columnData, projectedColumns, query, wasm);
    const outputColumns = query.projections.length > 0 ? query.projections : projectedColumns.map(c => c.name);
    if (query.projections.length > 0) {
      const keep = new Set(query.projections);
      for (const row of rows) {
        for (const key of Object.keys(row)) {
          if (!keep.has(key)) delete row[key];
        }
      }
    }

    return {
      rows,
      rowCount: rows.length,
      columns: outputColumns,
      bytesRead,
      pagesSkipped,
      durationMs: Date.now() - startTime,
      cacheHit: false,
    };
  }

  /** Execute a join query using HashJoinOperator. */
  private async executeJoin(
    query: QueryDescriptor,
    leftFragments: FragmentSource[],
    wasm: WasmEngine,
    pipeOpts: PipelineOptions | undefined,
    startTime: number,
  ): Promise<QueryResult> {
    const join = query.join!;

    // Build left pipeline (no sort/limit/agg — those apply after join)
    const leftQuery: QueryDescriptor = {
      table: query.table,
      filters: query.filters,
      projections: [], // fetch all columns for join
      join: undefined,
    };
    const leftScan = new (await import("./operators.js")).ScanOperator(leftFragments, leftQuery, wasm);
    let leftPipeline: import("./operators.js").Operator = leftScan;
    if (leftQuery.filters.length > 0) {
      leftPipeline = new (await import("./operators.js")).FilterOperator(leftPipeline, leftQuery.filters);
    }

    // Build right side — execute the right query to get a fragment source
    const rightIsUrl = join.right.table.startsWith("http://") || join.right.table.startsWith("https://");
    const rightMeta = await this.getOrLoadMeta(join.right.table);
    const rightNeeded = new Set(
      join.right.projections.length > 0 ? join.right.projections : rightMeta.columns.map(c => c.name),
    );
    for (const f of join.right.filters) rightNeeded.add(f.column);
    rightNeeded.add(join.rightKey);
    const rightProjected = rightMeta.columns.filter(c => rightNeeded.has(c.name));
    const rightFragment = await this.makeFragmentSource(join.right.table, rightProjected, rightIsUrl);

    const rightScan = new (await import("./operators.js")).ScanOperator([rightFragment], join.right, wasm);
    let rightPipeline: import("./operators.js").Operator = rightScan;
    if (join.right.filters.length > 0) {
      rightPipeline = new (await import("./operators.js")).FilterOperator(rightPipeline, join.right.filters);
    }

    // Hash join: right is build side, left is probe side
    const { HashJoinOperator, FsSpillBackend: FsSpillJoin } = await import("./operators.js");
    const joinSpill = pipeOpts?.spill ?? new FsSpillJoin();
    const joinBudget = pipeOpts?.memoryBudgetBytes ?? DEFAULT_MEMORY_BUDGET;
    const joinType = join.type ?? "inner";
    let pipeline: import("./operators.js").Operator = new HashJoinOperator(
      leftPipeline, rightPipeline, join.leftKey, join.rightKey, joinType,
      joinBudget, joinSpill,
    );

    // Apply post-join sort/limit/aggregates
    const { buildPipeline: _, TopKOperator, InMemorySortOperator, LimitOperator, ProjectOperator, AggregateOperator, ExternalSortOperator } = await import("./operators.js");
    const hasAgg = query.aggregates && query.aggregates.length > 0;

    if (hasAgg) {
      pipeline = new AggregateOperator(pipeline, query);
      if (query.sortColumn && query.limit !== undefined) {
        pipeline = new TopKOperator(pipeline, query.sortColumn, query.sortDirection === "desc", query.limit, query.offset ?? 0);
      } else if (query.sortColumn) {
        pipeline = new InMemorySortOperator(pipeline, query.sortColumn, query.sortDirection === "desc", query.offset ?? 0);
      } else if (query.offset || query.limit !== undefined) {
        pipeline = new LimitOperator(pipeline, query.limit ?? Infinity, query.offset ?? 0);
      }
    } else if (query.sortColumn) {
      if (query.limit !== undefined) {
        pipeline = new TopKOperator(pipeline, query.sortColumn, query.sortDirection === "desc", query.limit, query.offset ?? 0);
      } else {
        pipeline = new ExternalSortOperator(pipeline, query.sortColumn, query.sortDirection === "desc", query.offset ?? 0, pipeOpts?.memoryBudgetBytes ?? DEFAULT_MEMORY_BUDGET);
      }
    } else if (query.offset || query.limit !== undefined) {
      pipeline = new LimitOperator(pipeline, query.limit ?? Infinity, query.offset ?? 0);
    }

    // Project
    if (query.projections.length > 0) {
      pipeline = new ProjectOperator(pipeline, query.projections);
    }

    const rows = await drainPipeline(pipeline);

    return {
      rows,
      rowCount: rows.length,
      columns: query.projections.length > 0
        ? query.projections
        : [...new Set([...Object.keys(rows[0] ?? {})])],
      bytesRead: leftScan.bytesRead + rightScan.bytesRead,
      pagesSkipped: leftScan.pagesSkipped + rightScan.pagesSkipped,
      durationMs: Date.now() - startTime,
      cacheHit: false,
    };
  }

  /** List all versions of a Lance dataset. */
  async versions(table: string): Promise<VersionInfo[]> {
    const fs = await import("node:fs/promises");
    const pathMod = await import("node:path");

    const versionsDir = pathMod.join(table, "_versions");
    const entries = await fs.readdir(versionsDir).catch(() => [] as string[]);
    const manifests = entries.filter(e => e.endsWith(".manifest"))
      .sort((a, b) => parseInt(a, 10) - parseInt(b, 10));

    const results: VersionInfo[] = [];
    for (const manifestFile of manifests) {
      const manifestBuf = await fs.readFile(pathMod.join(versionsDir, manifestFile));
      const ab = manifestBuf.buffer.slice(manifestBuf.byteOffset, manifestBuf.byteOffset + manifestBuf.byteLength);
      const manifest = parseManifest(ab);
      if (!manifest) continue;

      const stat = await fs.stat(pathMod.join(versionsDir, manifestFile));
      results.push({
        version: manifest.version,
        timestamp: stat.mtimeMs,
        rowCount: manifest.totalRows,
        fragmentCount: manifest.fragments.length,
      });
    }

    return results.sort((a, b) => a.version - b.version);
  }

  /** Compute diff between two versions of a Lance dataset. */
  async diff(table: string, fromVersion: number, toVersion: number): Promise<DiffResult> {
    const fs = await import("node:fs/promises");
    const pathMod = await import("node:path");

    const versionsDir = pathMod.join(table, "_versions");

    const loadManifestFragments = async (version: number): Promise<{ paths: Set<string>; manifest: import("./types.js").ManifestInfo }> => {
      const manifestFile = `${version}.manifest`;
      const manifestBuf = await fs.readFile(pathMod.join(versionsDir, manifestFile));
      const ab = manifestBuf.buffer.slice(manifestBuf.byteOffset, manifestBuf.byteOffset + manifestBuf.byteLength);
      const manifest = parseManifest(ab);
      if (!manifest) throw new Error(`Failed to parse manifest for version ${version}`);
      return { paths: new Set(manifest.fragments.map(f => f.filePath)), manifest };
    };

    const from = await loadManifestFragments(fromVersion);
    const to = await loadManifestFragments(toVersion);

    const addedFragments: Set<string> = new Set();
    const removedFragments: Set<string> = new Set();

    for (const frag of to.paths) {
      if (!from.paths.has(frag)) addedFragments.add(frag);
    }
    for (const frag of from.paths) {
      if (!to.paths.has(frag)) removedFragments.add(frag);
    }

    let added = 0;
    for (const frag of to.manifest.fragments) {
      if (addedFragments.has(frag.filePath)) added += frag.physicalRows;
    }
    let removed = 0;
    for (const frag of from.manifest.fragments) {
      if (removedFragments.has(frag.filePath)) removed += frag.physicalRows;
    }

    return { added, removed, addedFragments: [...addedFragments], removedFragments: [...removedFragments] };
  }

  /** Load or retrieve cached dataset metadata for a Lance directory. */
  private async getOrLoadDataset(table: string, version?: number): Promise<DatasetMeta> {
    const cacheKey = version != null ? `${table}@v${version}` : table;
    let dataset = this.datasetCache.get(cacheKey);
    if (dataset) return dataset;

    const fs = await import("node:fs/promises");
    const pathMod = await import("node:path");

    const versionsDir = pathMod.join(table, "_versions");
    const entries = await fs.readdir(versionsDir).catch(() => [] as string[]);
    const manifests = entries.filter(e => e.endsWith(".manifest"))
      .sort((a, b) => parseInt(a, 10) - parseInt(b, 10));
    if (manifests.length === 0) {
      throw new Error(`No manifests found in ${versionsDir}`);
    }

    let targetManifest: string;
    if (version != null) {
      targetManifest = `${version}.manifest`;
      if (!manifests.includes(targetManifest)) {
        throw new Error(`Version ${version} not found in ${versionsDir}. Available: ${manifests.join(", ")}`);
      }
    } else {
      targetManifest = manifests[manifests.length - 1];
    }
    const manifestBuf = await fs.readFile(pathMod.join(versionsDir, targetManifest));
    const ab = manifestBuf.buffer.slice(manifestBuf.byteOffset, manifestBuf.byteOffset + manifestBuf.byteLength);
    const manifest = parseManifest(ab);
    if (!manifest) throw new Error(`Failed to parse manifest ${targetManifest}`);

    const fragmentMetas = new Map<number, TableMeta>();
    for (const frag of manifest.fragments) {
      let fragPath = pathMod.join(table, frag.filePath);
      try { await fs.stat(fragPath); } catch {
        fragPath = pathMod.join(table, "data", frag.filePath);
      }
      try {
        const cachedMeta = await this.loadMetaFromFile(fragPath);
        let { columns } = cachedMeta;

        const isLanceV2 = columns.some(c => c.name.startsWith("column_")) || columns.every(c => c.dtype === "int64");
        if (isLanceV2 || !columns.some(c => c.pages.length > 0)) {
          const fragBuf = await fs.readFile(fragPath);
          const fragAb = fragBuf.buffer.slice(fragBuf.byteOffset, fragBuf.byteOffset + fragBuf.byteLength);
          const v2Cols = parseLanceV2Columns(fragAb, manifest.schema, frag.physicalRows);
          if (v2Cols && v2Cols.length > 0) {
            columns = lanceV2ToColumnMeta(v2Cols);
            // Compute and cache min/max stats for page skipping
            const filePath = fragPath;
            await computeLanceV2Stats(columns, async (offset, length) => {
              const handle = await fs.open(filePath, "r");
              try {
                const buf = Buffer.alloc(length);
                await handle.read(buf, 0, length, offset);
                return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
              } finally {
                await handle.close();
              }
            });
          }
        }

        const fragStat = await fs.stat(fragPath);
        const fragHandle = await fs.open(fragPath, "r");
        let footer: import("./types.js").Footer;
        try {
          const footerBuf = Buffer.alloc(FOOTER_SIZE);
          await fragHandle.read(footerBuf, 0, FOOTER_SIZE, fragStat.size - FOOTER_SIZE);
          const footerAb = footerBuf.buffer.slice(footerBuf.byteOffset, footerBuf.byteOffset + footerBuf.byteLength);
          const parsed = parseFooter(footerAb);
          if (!parsed) continue;
          footer = parsed;
        } finally {
          await fragHandle.close();
        }
        fragmentMetas.set(frag.id, {
          name: frag.filePath,
          footer,
          columns,
          totalRows: frag.physicalRows,
          fileSize: BigInt(cachedMeta.fileSize),
          r2Key: fragPath,
          updatedAt: Date.now(),
        });
      } catch (err) {
        throw new Error(`Failed to load fragment ${frag.filePath}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    dataset = {
      name: table,
      r2Prefix: table + "/",
      manifest,
      fragmentMetas,
      totalRows: manifest.totalRows,
      updatedAt: Date.now(),
    };
    this.datasetCache.set(cacheKey, dataset);
    if (this.datasetCache.size > 100) {
      const firstKey = this.datasetCache.keys().next().value;
      if (firstKey) this.datasetCache.delete(firstKey);
    }
    return dataset;
  }

  /** Execute a query against a multi-fragment Lance dataset directory. */
  private async executeDatasetQuery(query: QueryDescriptor, startTime: number): Promise<QueryResult> {
    const metaStart = Date.now();
    const dataset = await this.getOrLoadDataset(query.table, query.version);

    // Validate column references against schema
    const schemaColumnNames = new Set(dataset.manifest.schema.map(f => f.name));
    for (const p of query.projections) {
      if (!schemaColumnNames.has(p)) {
        throw new QueryModeError("COLUMN_NOT_FOUND", `Column "${p}" not found in ${query.table}. Available: ${[...schemaColumnNames].join(", ")}`);
      }
    }
    for (const f of query.filters) {
      if (!schemaColumnNames.has(f.column)) {
        throw new QueryModeError("COLUMN_NOT_FOUND", `Filter column "${f.column}" not found in ${query.table}. Available: ${[...schemaColumnNames].join(", ")}`);
      }
    }
    if (query.sortColumn && !schemaColumnNames.has(query.sortColumn)) {
      const aggAliases = new Set((query.aggregates ?? []).map(a => a.alias ?? `${a.fn}_${a.column}`));
      const groupCols = new Set(query.groupBy ?? []);
      if (!aggAliases.has(query.sortColumn) && !groupCols.has(query.sortColumn)) {
        throw new QueryModeError("COLUMN_NOT_FOUND", `Sort column "${query.sortColumn}" not found in ${query.table}. Available: ${[...schemaColumnNames].join(", ")}`);
      }
    }

    const metaMs = Date.now() - metaStart;

    // Build fragment sources for all fragments
    const fsMod = await import("node:fs/promises");
    const wasm = await this.getWasm();
    const fragments: FragmentSource[] = [];

    for (const [, meta] of dataset.fragmentMetas) {
      // Skip entire fragment if min/max stats eliminate it
      if (canSkipFragment(meta, query.filters, query.filterGroups)) continue;

      const neededCols = queryReferencedColumns(query, meta.columns.map(c => c.name));
      const projectedColumns = meta.columns.filter(c => neededCols.has(c.name));
      const filePath = meta.r2Key;

      fragments.push({
        columns: projectedColumns,
        async readPage(_col: ColumnMeta, page: PageInfo): Promise<ArrayBuffer> {
          const handle = await fsMod.open(filePath, "r");
          try {
            const buf = Buffer.alloc(page.byteLength);
            await handle.read(buf, 0, page.byteLength, Number(page.byteOffset));
            return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
          } finally {
            await handle.close();
          }
        },
      });
    }

    const outputColumns = query.projections.length > 0
      ? query.projections
      : (dataset.fragmentMetas.values().next().value?.columns.map((c: ColumnMeta) => c.name) ?? []);

    const { FsSpillBackend: FsSpill } = await import("./operators.js");
    const pipeOpts: PipelineOptions = {
      memoryBudgetBytes: this.memoryBudgetBytes ?? DEFAULT_MEMORY_BUDGET,
      spill: new FsSpill(),
    };
    const { pipeline, scan, wasmAgg } = buildPipeline(fragments, query, wasm, outputColumns, pipeOpts);
    const rows = await drainPipeline(pipeline);

    const result: QueryResult & { scanMs?: number; pipelineMs?: number; metaMs?: number } = {
      rows,
      rowCount: rows.length,
      columns: outputColumns,
      bytesRead: wasmAgg?.bytesRead ?? scan?.bytesRead ?? 0,
      pagesSkipped: wasmAgg?.pagesSkipped ?? scan?.pagesSkipped ?? 0,
      durationMs: Date.now() - startTime,
      scanMs: wasmAgg?.scanMs ?? scan?.scanMs ?? 0,
      pipelineMs: Date.now() - startTime - metaMs,
      metaMs,
    };
    return result;
  }

  /** Load footer + column metadata from a local file */
  private async loadMetaFromFile(path: string): Promise<{ columns: ColumnMeta[]; fileSize: number }> {
    const fs = await import("node:fs/promises");
    const stat = await fs.stat(path).catch((err: unknown) => {
      throw QueryModeError.from(err, { table: path });
    });
    const fileSize = Number(stat.size);

    const handle = await fs.open(path, "r");
    try {
      // Read last 40 bytes for format detection
      const tailSize = Math.min(fileSize, FOOTER_SIZE);
      const tailBuf = Buffer.alloc(tailSize);
      await handle.read(tailBuf, 0, tailSize, fileSize - tailSize);
      const tailAb = tailBuf.buffer.slice(tailBuf.byteOffset, tailBuf.byteOffset + tailBuf.byteLength);

      const fmt = detectFormat(tailAb);

      if (fmt === "parquet") {
        const footerLen = getParquetFooterLength(tailAb);
        if (!footerLen) throw new QueryModeError("INVALID_FORMAT", `Invalid Parquet file: cannot read footer length in ${path}`);

        const parquetFooterBuf = Buffer.alloc(footerLen);
        await handle.read(parquetFooterBuf, 0, footerLen, fileSize - footerLen - 8);
        const parquetFooterAb = parquetFooterBuf.buffer.slice(
          parquetFooterBuf.byteOffset, parquetFooterBuf.byteOffset + parquetFooterBuf.byteLength,
        );

        const parquetMeta = parseParquetFooter(parquetFooterAb);
        if (!parquetMeta) throw new QueryModeError("INVALID_FORMAT", `Failed to parse Parquet footer in ${path}`);

        const tableMeta = parquetMetaToTableMeta(parquetMeta, path, BigInt(fileSize));
        return { columns: tableMeta.columns, fileSize };
      }

      // Lance format (default) — but try ReaderRegistry first for non-Lance/Parquet formats
      const footer = parseFooter(tailAb);
      if (!footer) {
        // Not Lance or Parquet — try pluggable readers (CSV, JSON, Arrow IPC, etc.)
        const registry = await this.getReaderRegistry();
        const ext = path.substring(path.lastIndexOf("."));
        let reader = registry.getByExtension(ext);
        if (!reader) {
          const { FileDataSource } = await import("./reader.js");
          const source = new FileDataSource(path);
          reader = await registry.detect(source);
        }
        if (reader) {
          const { FileDataSource } = await import("./reader.js");
          const source = new FileDataSource(path);
          const meta = await reader.readMeta(source);
          // Cache fragment sources for later use in execute()
          const fragments = await reader.createFragments(source, meta.columns);
          this.readerFragmentCache.set(path, fragments);
          return { columns: meta.columns, fileSize };
        }
        throw new QueryModeError("INVALID_FORMAT", `Invalid file format: unrecognized magic in ${path}. Supported formats: .lance, .parquet, .csv, .tsv, .json, .ndjson, .jsonl, .arrow, .ipc, .feather`);
      }

      // Read column metadata (protobuf region)
      const metaStart = Number(footer.columnMetaStart);
      const metaEnd = Number(footer.columnMetaOffsetsStart);
      const metaLength = metaEnd - metaStart;

      if (metaLength <= 0) return { columns: [], fileSize };

      const metaBuf = Buffer.alloc(metaLength);
      await handle.read(metaBuf, 0, metaLength, metaStart);
      const metaAb = metaBuf.buffer.slice(metaBuf.byteOffset, metaBuf.byteOffset + metaBuf.byteLength);

      const columns = parseColumnMetaFromProtobuf(metaAb, footer.numColumns);

      // If standard parser returned 0 pages (Lance v2), try v2 parser on full file
      const hasPages = columns.some(c => c.pages.length > 0);
      if (!hasPages && footer.numColumns > 0) {
        const fullBuf = Buffer.alloc(fileSize);
        await handle.read(fullBuf, 0, fileSize, 0);
        const fullAb = fullBuf.buffer.slice(fullBuf.byteOffset, fullBuf.byteOffset + fullBuf.byteLength);
        const v2Cols = parseLanceV2Columns(fullAb);
        if (v2Cols && v2Cols.length > 0) {
          const v2Columns = lanceV2ToColumnMeta(v2Cols);
          // Compute min/max stats for page skipping
          const fileHandle = handle;
          await computeLanceV2Stats(v2Columns, async (offset, length) => {
            const buf = Buffer.alloc(length);
            await fileHandle.read(buf, 0, length, offset);
            return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
          });
          return { columns: v2Columns, fileSize };
        }
      }

      return { columns, fileSize };
    } finally {
      await handle.close();
    }
  }

  /** Load footer + column metadata via HTTP Range requests */
  private async loadMetaFromUrl(url: string): Promise<{ columns: ColumnMeta[]; fileSize: number }> {
    // Get file size
    const headResp = await fetch(url, { method: "HEAD" });
    const fileSize = Number(headResp.headers.get("content-length") ?? 0);
    if (fileSize < 8) throw new Error(`File too small: ${url}`);

    // Read tail for format detection (last 40 bytes)
    const tailSize = Math.min(fileSize, FOOTER_SIZE);
    const tailStart = fileSize - tailSize;
    const tailResp = await fetch(url, {
      headers: { Range: `bytes=${tailStart}-${fileSize - 1}` },
    });
    const tailAb = await tailResp.arrayBuffer();
    const fmt = detectFormat(tailAb);

    if (fmt === "parquet") {
      const footerLen = getParquetFooterLength(tailAb);
      if (!footerLen) throw new QueryModeError("INVALID_FORMAT", `Invalid Parquet file: cannot read footer length in ${url}`);

      const footerStart = fileSize - footerLen - 8;
      const footerResp = await fetch(url, {
        headers: { Range: `bytes=${footerStart}-${footerStart + footerLen - 1}` },
      });
      const footerBuf = await footerResp.arrayBuffer();
      const parquetMeta = parseParquetFooter(footerBuf);
      if (!parquetMeta) throw new QueryModeError("INVALID_FORMAT", `Failed to parse Parquet footer in ${url}`);

      const tableMeta = parquetMetaToTableMeta(parquetMeta, url, BigInt(fileSize));
      return { columns: tableMeta.columns, fileSize };
    }

    // Lance format
    const footer = parseFooter(tailAb);
    if (!footer) throw new QueryModeError("INVALID_FORMAT", `Invalid file format: unrecognized magic in ${url}. Supported formats: .lance, .parquet, .csv, .tsv, .json, .ndjson, .jsonl, .arrow, .ipc, .feather`);

    const metaStart = Number(footer.columnMetaStart);
    const metaEnd = Number(footer.columnMetaOffsetsStart);
    const metaLength = metaEnd - metaStart;

    if (metaLength <= 0) return { columns: [], fileSize };

    const metaResp = await fetch(url, {
      headers: { Range: `bytes=${metaStart}-${metaStart + metaLength - 1}` },
    });
    const metaAb = await metaResp.arrayBuffer();
    const columns = parseColumnMetaFromProtobuf(metaAb, footer.numColumns);

    return { columns, fileSize };
  }
}
