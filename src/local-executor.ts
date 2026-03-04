/**
 * LocalExecutor — query execution on local filesystem (Node/Bun).
 *
 * Extracted from index.ts so that local.ts can import it without
 * triggering Cloudflare DO module loads (wasm-module.ts uses Wrangler
 * .wasm import syntax that crashes in Node).
 */
import type { QueryDescriptor, QueryExecutor } from "./client.js";
import type { AppendResult, ColumnMeta, DataType, ExplainResult, QueryResult, Row, TableMeta, DatasetMeta } from "./types.js";
import { parseFooter, parseColumnMetaFromProtobuf, FOOTER_SIZE } from "./footer.js";
import { parseManifest } from "./manifest.js";
import { detectFormat, getParquetFooterLength, parseParquetFooter, parquetMetaToTableMeta } from "./parquet.js";
import { assembleRows, canSkipPage } from "./decode.js";
import { coalesceRanges } from "./coalesce.js";
import { instantiateWasm, type WasmEngine } from "./wasm-engine.js";
import { computePartialAgg, finalizePartialAgg } from "./partial-agg.js";
import { VipCache } from "./vip-cache.js";
import { QueryModeError } from "./errors.js";
import { parseLanceV2Columns, lanceV2ToColumnMeta } from "./lance-v2.js";

/**
 * Executor for local mode (Node/Bun).
 * Reads Lance/Parquet files directly from the filesystem or via HTTP.
 * Footer is parsed on first access and cached in-process.
 */
export class LocalExecutor implements QueryExecutor {
  private metaCache: Map<string, { columns: ColumnMeta[]; fileSize: number }> = new Map();
  private datasetCache: Map<string, DatasetMeta> = new Map();
  private resultCache = new VipCache<string, QueryResult>(200, 2);
  private wasmModule?: WebAssembly.Module;
  private wasmEngine?: WasmEngine;

  constructor(wasmModule?: WebAssembly.Module) {
    this.wasmModule = wasmModule;
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
          for (let i = 0; i < rows.length; i++) arr[i] = BigInt(rows[i][colName] as number);
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
        const enc = new TextEncoder();
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

    // Invalidate meta cache
    this.metaCache.delete(tablePath);
    this.datasetCache.delete(tablePath);

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
    if (query.filters.length === 0) {
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
    const projectedColumns = query.projections.length > 0
      ? columns.filter(c => query.projections.includes(c.name))
      : columns;

    let pagesTotal = 0;
    let pagesSkipped = 0;
    let estimatedRows = 0;
    const ranges: { column: string; offset: number; length: number }[] = [];
    const colDetails: ExplainResult["columns"] = [];

    // Use first column for row estimation (all columns have same page structure)
    const firstCol = projectedColumns[0];
    for (const col of projectedColumns) {
      let colBytes = 0;
      let colPages = 0;
      for (let pi = 0; pi < col.pages.length; pi++) {
        const page = col.pages[pi];
        pagesTotal++;
        if (!query.vectorSearch && canSkipPage(page, query.filters, col.name)) {
          pagesSkipped++;
          continue;
        }
        colPages++;
        colBytes += page.byteLength;
        ranges.push({ column: col.name, offset: Number(page.byteOffset), length: page.byteLength });
        // Count estimated rows from first projected column's non-skipped pages
        if (col === firstCol) estimatedRows += page.rowCount;
      }
      colDetails.push({ name: col.name, dtype: col.dtype as DataType, pages: colPages, bytes: colBytes });
    }

    const coalesced = coalesceRanges(ranges, 64 * 1024);
    const estimatedBytes = ranges.reduce((s, r) => s + r.length, 0);
    const totalRows = columns[0]?.pages.reduce((s, p) => s + p.rowCount, 0) ?? 0;

    // Detect format from file
    const format = await this.detectFileFormat(query.table);

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
      fragments: 1,
      filters: query.filters.map(f => ({
        column: f.column,
        op: f.op,
        pushable: f.op !== "in" && f.op !== "neq",
      })),
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
    const meta = await this.getOrLoadMeta(query.table);
    const { columns } = meta;
    const projectedColumns = query.projections.length > 0
      ? columns.filter(c => query.projections.includes(c.name))
      : columns;

    const isUrl = query.table.startsWith("http://") || query.table.startsWith("https://");
    const firstCol = projectedColumns[0];
    if (!firstCol) return;

    // If sorted, must buffer all rows then chunk
    if (query.sortColumn) {
      const result = await this.execute(query);
      for (let i = 0; i < result.rows.length; i += batchSize) {
        yield result.rows.slice(i, i + batchSize);
      }
      return;
    }

    const totalPages = firstCol.pages.length;
    let pageIdx = 0;
    let totalYielded = 0;

    const fs = isUrl ? null : await import("node:fs/promises");
    const handle = isUrl ? null : await fs!.open(query.table, "r");
    const wasm = await this.getWasm();

    try {
      while (pageIdx < totalPages) {
        let batchRows = 0;
        const batchStartPage = pageIdx;
        while (pageIdx < totalPages && batchRows < batchSize) {
          const page = firstCol.pages[pageIdx];
          if (!query.vectorSearch && canSkipPage(page, query.filters, firstCol.name)) {
            pageIdx++;
            continue;
          }
          batchRows += page.rowCount;
          pageIdx++;
        }

        if (batchRows === 0) continue;

        const columnData = new Map<string, ArrayBuffer[]>();
        for (const col of projectedColumns) {
          for (let pi = batchStartPage; pi < pageIdx; pi++) {
            const page = col.pages[pi];
            if (!page) continue;
            if (!query.vectorSearch && canSkipPage(page, query.filters, col.name)) continue;

            let ab: ArrayBuffer;
            if (isUrl) {
              const start = Number(page.byteOffset);
              const end = start + page.byteLength - 1;
              const resp = await fetch(query.table, { headers: { Range: `bytes=${start}-${end}` } });
              ab = await resp.arrayBuffer();
            } else {
              const buf = Buffer.alloc(page.byteLength);
              await handle!.read(buf, 0, page.byteLength, Number(page.byteOffset));
              ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
            }

            const arr = columnData.get(col.name) ?? [];
            arr.push(ab);
            columnData.set(col.name, arr);
          }
        }

        const rows = assembleRows(columnData, projectedColumns, query, wasm);

        // Chunk rows into batchSize-sized batches
        for (let ri = 0; ri < rows.length; ri += batchSize) {
          const chunk = rows.slice(ri, ri + batchSize);
          if (query.limit && totalYielded + chunk.length > query.limit) {
            yield chunk.slice(0, query.limit - totalYielded);
            return;
          }
          yield chunk;
          totalYielded += chunk.length;
        }
      }
    } finally {
      if (handle) await handle.close();
    }
  }

  /** Get or load table metadata (footer + columns). Caches in-memory. */
  private async getOrLoadMeta(table: string): Promise<{ columns: ColumnMeta[]; fileSize: number }> {
    let cached = this.metaCache.get(table);
    if (cached) return cached;
    const isUrl = table.startsWith("http://") || table.startsWith("https://");
    cached = isUrl ? await this.loadMetaFromUrl(table) : await this.loadMetaFromFile(table);
    this.metaCache.set(table, cached);
    if (this.metaCache.size > 1000) {
      const firstKey = this.metaCache.keys().next().value;
      if (firstKey) this.metaCache.delete(firstKey);
    }
    return cached;
  }

  /** Build a cache key from query descriptor. */
  private queryCacheKey(query: QueryDescriptor): string {
    const { table, filters, projections, sortColumn, sortDirection, limit, offset, aggregates, groupBy } = query;
    return `qr:${table}:${JSON.stringify({ filters, projections, sortColumn, sortDirection, limit, offset, aggregates, groupBy })}`;
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
      throw new QueryModeError("COLUMN_NOT_FOUND", `Sort column "${query.sortColumn}" not found in ${query.table}. Available: ${[...columnNames].join(", ")}`);
    }

    // Step 2: Determine projected columns
    const projectedColumns =
      query.projections.length > 0
        ? columns.filter((c) => query.projections.includes(c.name))
        : columns;

    // Step 3: Determine which pages to fetch using filter pushdown
    const pageRanges: { column: string; offset: bigint; length: number }[] = [];
    let pagesSkipped = 0;

    for (const col of projectedColumns) {
      for (const page of col.pages) {
        if (!query.vectorSearch && canSkipPage(page, query.filters, col.name)) {
          pagesSkipped++;
          continue;
        }
        pageRanges.push({ column: col.name, offset: page.byteOffset, length: page.byteLength });
      }
    }

    // Step 4: Read page data
    const columnData = new Map<string, ArrayBuffer[]>();
    let bytesRead = 0;

    if (isUrl) {
      // Parallel HTTP Range reads
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
      // Sequential reads from local filesystem
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

    // Step 5: Decode and assemble rows
    const wasm = await this.getWasm();
    const hasAggregates = query.aggregates && query.aggregates.length > 0;
    // When aggregating, defer sort/limit — they apply to aggregated output, not raw rows
    const assembleQuery = hasAggregates
      ? { ...query, sortColumn: undefined, sortDirection: undefined, limit: undefined, offset: undefined }
      : query;
    let rows = assembleRows(columnData, projectedColumns, assembleQuery, wasm);

    // Step 6: Apply aggregation if requested, then sort/limit the result
    if (hasAggregates) {
      const partial = computePartialAgg(rows, query);
      rows = finalizePartialAgg(partial, query);
      // Apply sort and limit to aggregated rows
      if (query.sortColumn) {
        const dir = query.sortDirection === "desc" ? -1 : 1;
        const sc = query.sortColumn;
        rows.sort((a, b) => {
          const av = a[sc], bv = b[sc];
          if (av == null && bv == null) return 0;
          if (av == null) return 1;
          if (bv == null) return -1;
          return av < bv ? -dir : av > bv ? dir : 0;
        });
      }
      const offset = query.offset ?? 0;
      if (offset > 0 || query.limit !== undefined) {
        rows = rows.slice(offset, query.limit !== undefined ? offset + query.limit : undefined);
      }
    }

    const result: QueryResult = {
      rows,
      rowCount: rows.length,
      columns: projectedColumns.map((c) => c.name),
      bytesRead,
      pagesSkipped,
      durationMs: Date.now() - startTime,
      cacheHit: false,
    };

    // Store in result cache if TTL specified
    if (query.cacheTTL && !query.vectorSearch) {
      this.resultCache.setWithTTL(this.queryCacheKey(query), result, query.cacheTTL);
    }

    return result;
  }

  /** Execute a query against a multi-fragment Lance dataset directory. */
  private async executeDatasetQuery(query: QueryDescriptor, startTime: number): Promise<QueryResult> {
    const fs = await import("node:fs/promises");
    const pathMod = await import("node:path");

    // Discover or reuse cached dataset metadata
    let dataset = this.datasetCache.get(query.table);
    if (!dataset) {
      // Find latest manifest
      const versionsDir = pathMod.join(query.table, "_versions");
      const entries = await fs.readdir(versionsDir).catch(() => [] as string[]);
      const manifests = entries.filter(e => e.endsWith(".manifest")).sort();
      if (manifests.length === 0) {
        throw new Error(`No manifests found in ${versionsDir}`);
      }

      const latestManifest = manifests[manifests.length - 1];
      const manifestBuf = await fs.readFile(pathMod.join(versionsDir, latestManifest));
      const ab = manifestBuf.buffer.slice(manifestBuf.byteOffset, manifestBuf.byteOffset + manifestBuf.byteLength);
      const manifest = parseManifest(ab);
      if (!manifest) throw new Error(`Failed to parse manifest ${latestManifest}`);

      // Read footer + column metadata for each fragment
      const fragmentMetas = new Map<number, TableMeta>();
      for (const frag of manifest.fragments) {
        // Lance stores relative paths without data/ prefix — try both
        let fragPath = pathMod.join(query.table, frag.filePath);
        try { await fs.stat(fragPath); } catch {
          fragPath = pathMod.join(query.table, "data", frag.filePath);
        }
        try {
          const cachedMeta = await this.loadMetaFromFile(fragPath);
          let { columns } = cachedMeta;

          // Always try v2 parser with manifest schema for Lance files.
          // The schema provides correct column names and types that the
          // no-schema fallback in loadMetaFromFile cannot determine.
          const isLanceV2 = columns.some(c => c.name.startsWith("column_")) || columns.every(c => c.dtype === "int64");
          if (isLanceV2 || !columns.some(c => c.pages.length > 0)) {
            const fragBuf = await fs.readFile(fragPath);
            const fragAb = fragBuf.buffer.slice(fragBuf.byteOffset, fragBuf.byteOffset + fragBuf.byteLength);
            const v2Cols = parseLanceV2Columns(fragAb, manifest.schema, frag.physicalRows);
            if (v2Cols && v2Cols.length > 0) {
              columns = lanceV2ToColumnMeta(v2Cols);
            }
          }

          // Read the actual footer from the fragment file
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
        name: query.table,
        r2Prefix: query.table + "/",
        manifest,
        fragmentMetas,
        totalRows: manifest.totalRows,
        updatedAt: Date.now(),
      };
      this.datasetCache.set(query.table, dataset);
      if (this.datasetCache.size > 100) {
        const firstKey = this.datasetCache.keys().next().value;
        if (firstKey) this.datasetCache.delete(firstKey);
      }
    }

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
      throw new QueryModeError("COLUMN_NOT_FOUND", `Sort column "${query.sortColumn}" not found in ${query.table}. Available: ${[...schemaColumnNames].join(", ")}`);
    }

    // Execute query across all fragments
    const fsMod = await import("node:fs/promises");
    let allRows: Row[] = [];
    let totalBytesRead = 0;
    let totalPagesSkipped = 0;

    for (const [, meta] of dataset.fragmentMetas) {
      const projectedColumns = query.projections.length > 0
        ? meta.columns.filter(c => query.projections.includes(c.name))
        : meta.columns;

      const pageRanges: { column: string; offset: bigint; length: number }[] = [];
      for (const col of projectedColumns) {
        for (const page of col.pages) {
          if (!query.vectorSearch && canSkipPage(page, query.filters, col.name)) {
            totalPagesSkipped++;
            continue;
          }
          pageRanges.push({ column: col.name, offset: page.byteOffset, length: page.byteLength });
        }
      }

      const columnData = new Map<string, ArrayBuffer[]>();
      const handle = await fsMod.open(meta.r2Key, "r");
      try {
        for (const range of pageRanges) {
          const buf = Buffer.alloc(range.length);
          await handle.read(buf, 0, range.length, Number(range.offset));
          totalBytesRead += range.length;
          const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
          const existing = columnData.get(range.column) ?? [];
          existing.push(ab);
          columnData.set(range.column, existing);
        }
      } finally {
        await handle.close();
      }

      const wasm = await this.getWasm();
      const rows = assembleRows(columnData, projectedColumns, query, wasm);
      for (let ri = 0; ri < rows.length; ri++) allRows.push(rows[ri]);
    }

    // Apply limit if no sort and no aggregates
    const hasAgg = query.aggregates && query.aggregates.length > 0;
    if (query.limit && !query.sortColumn && !hasAgg && allRows.length > query.limit) {
      allRows = allRows.slice(0, query.limit);
    }

    // Apply aggregation if requested, then sort/limit the result
    if (hasAgg) {
      const partial = computePartialAgg(allRows, query);
      allRows = finalizePartialAgg(partial, query);
      if (query.sortColumn) {
        const dir = query.sortDirection === "desc" ? -1 : 1;
        const sc = query.sortColumn;
        allRows.sort((a, b) => {
          const av = a[sc], bv = b[sc];
          if (av == null && bv == null) return 0;
          if (av == null) return 1;
          if (bv == null) return -1;
          return av < bv ? -dir : av > bv ? dir : 0;
        });
      }
      const offset = query.offset ?? 0;
      if (offset > 0 || query.limit !== undefined) {
        allRows = allRows.slice(offset, query.limit !== undefined ? offset + query.limit : undefined);
      }
    }

    return {
      rows: allRows,
      rowCount: allRows.length,
      columns: query.projections.length > 0
        ? query.projections
        : (dataset.fragmentMetas.values().next().value?.columns.map((c: ColumnMeta) => c.name) ?? []),
      bytesRead: totalBytesRead,
      pagesSkipped: totalPagesSkipped,
      durationMs: Date.now() - startTime,
    };
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

      // Lance format (default)
      const footer = parseFooter(tailAb);
      if (!footer) throw new QueryModeError("INVALID_FORMAT", `Invalid file format: unrecognized magic in ${path}`);

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
          return { columns: lanceV2ToColumnMeta(v2Cols), fileSize };
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
    if (!footer) throw new QueryModeError("INVALID_FORMAT", `Invalid file format: unrecognized magic in ${url}`);

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
