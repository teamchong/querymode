import { DurableObject } from "cloudflare:workers";
import type { ColumnMeta, Env, Footer, TableMeta, DatasetMeta, AppendResult, AppendOptions, DropResult } from "./types.js";
import { NULL_SENTINEL } from "./types.js";
import { parseFooter, parseColumnMetaFromProtobuf, FOOTER_SIZE } from "./footer.js";
import { parseManifest } from "./manifest.js";
import { detectFormat, getParquetFooterLength, parseParquetFooter, parquetMetaToTableMeta } from "./parquet.js";
import type { QueryDORpc } from "./types.js";
import { instantiateWasm, rowsToColumnArrays, type WasmEngine } from "./wasm-engine.js";
import { resolveBucket } from "./bucket.js";
import { withTimeout } from "./coalesce.js";
import wasmModule from "./wasm-module.js";

const textEncoder = new TextEncoder();

/** Master DO — single writer, reads footers, broadcasts invalidations. */
export class MasterDO extends DurableObject<Env> {
  private broadcastFailures = new Map<string, number>(); // region → consecutive failure count
  private wasmEngine?: WasmEngine;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
  }

  // ── RPC methods ────────────────────────────────────────────────────────

  async registerRpc(queryDoId: string, region: string): Promise<{ registered: boolean; region: string; tableVersions?: Record<string, { r2Key: string; updatedAt: number }> }> {
    const regions = (await this.ctx.storage.get<Record<string, string>>("regions")) ?? {};
    regions[region] = queryDoId;
    await this.ctx.storage.put("regions", regions);

    // Return current table timestamps so waking Query DOs can detect stale caches
    const tables = await this.ctx.storage.list<TableMeta>({ prefix: "table:" });
    const tableVersions: Record<string, { r2Key: string; updatedAt: number }> = {};
    for (const [key, meta] of tables) {
      const name = key.replace("table:", "");
      tableVersions[name] = { r2Key: meta.r2Key ?? name, updatedAt: meta.updatedAt ?? 0 };
    }
    return { registered: true, region, tableVersions };
  }

  async writeRpc(body: unknown): Promise<unknown> {
    const { r2Key } = body as { r2Key: string };
    if (!r2Key || typeof r2Key !== "string" || r2Key.includes("..")) {
      throw new Error("Invalid r2Key");
    }

    // Check if this is a dataset directory (ends with / or .lance/)
    if (r2Key.endsWith("/") || r2Key.endsWith(".lance/")) {
      return this.executeDatasetWrite(r2Key);
    }

    const result = await this.readFooterAndColumns(r2Key);
    if (!result) throw new Error("Failed to read footer");

    const tableName = r2Key.replace(/\.(lance|parquet)$/, "").split("/").pop() ?? r2Key;
    const totalRows = result.columns[0]?.pages.reduce((s, p) => s + p.rowCount, 0) ?? 0;
    const meta: TableMeta = {
      name: tableName, footer: result.parsed, format: result.format, columns: result.columns,
      totalRows, fileSize: result.fileSize, r2Key, updatedAt: Date.now(),
    };
    await this.ctx.storage.put(`table:${tableName}`, meta);
    await this.broadcast(tableName, r2Key, result, { totalRows });
    return { success: true, table: tableName };
  }

  /** Write notification for a multi-fragment dataset directory. */
  private async executeDatasetWrite(r2Prefix: string): Promise<unknown> {
    const tableName = r2Prefix.replace(/\/$/, "").replace(/\.lance$/, "").split("/").pop() ?? r2Prefix;

    // Find latest manifest
    const listed = await resolveBucket(this.env, r2Prefix).list({ prefix: `${r2Prefix}_versions/`, limit: 100 });
    const manifestKeys = listed.objects
      .filter(o => o.key.endsWith(".manifest"))
      .sort((a, b) => { const na = parseInt(a.key.split("/").pop()!, 10); const nb = parseInt(b.key.split("/").pop()!, 10); return na - nb; });
    if (manifestKeys.length === 0) throw new Error("No manifests found");

    const latestKey = manifestKeys[manifestKeys.length - 1].key;
    const manifestObj = await resolveBucket(this.env, latestKey).get(latestKey);
    if (!manifestObj) throw new Error("Failed to read manifest");

    const manifest = parseManifest(await manifestObj.arrayBuffer());
    if (!manifest) throw new Error("Failed to parse manifest");

    // Read first fragment's footer to broadcast (Query DOs will discover the rest)
    if (manifest.fragments.length > 0) {
      const firstFrag = manifest.fragments[0];
      // Try filePath as-is, then with data/ prefix (Lance stores relative paths)
      let fragKey = `${r2Prefix}${firstFrag.filePath}`;
      let result = await this.readFooterAndColumns(fragKey);
      if (!result) {
        fragKey = `${r2Prefix}data/${firstFrag.filePath}`;
        result = await this.readFooterAndColumns(fragKey);
      }
      if (result) {
        await this.broadcast(tableName, fragKey, result, { totalRows: manifest.totalRows, r2Prefix });
      }
    }

    await this.ctx.storage.put(`table:${tableName}`, {
      name: tableName, r2Prefix, manifest, totalRows: manifest.totalRows, updatedAt: Date.now(),
    });

    return { success: true, table: tableName, fragments: manifest.fragments.length, totalRows: manifest.totalRows };
  }

  private async getWasm(): Promise<WasmEngine> {
    if (this.wasmEngine) return this.wasmEngine;
    this.wasmEngine = await instantiateWasm(wasmModule);
    return this.wasmEngine;
  }

  /** Core append logic. Supports partitioned writes via options.partitionBy. */
  private async executeAppend(table: string, rows: Record<string, unknown>[], options?: AppendOptions): Promise<AppendResult> {
    if (!rows?.length) throw new Error("No rows provided");

    // Partition-aware ingest: split rows by partition value, write separate fragments
    if (options?.partitionBy) {
      const partCol = options.partitionBy;
      const groups = new Map<string, Record<string, unknown>[]>();
      for (const row of rows) {
        const key = row[partCol] === null || row[partCol] === undefined ? NULL_SENTINEL : String(row[partCol]);
        let group = groups.get(key);
        if (!group) { group = []; groups.set(key, group); }
        group.push(row);
      }

      // Write each partition group as a separate fragment
      let totalWritten = 0;
      let lastResult: AppendResult | null = null;
      for (const [, groupRows] of groups) {
        lastResult = await this.executeAppendSingle(table, groupRows, options);
        totalWritten += groupRows.length;
      }

      return {
        ...lastResult!,
        rowsWritten: totalWritten,
        metadata: { ...options.metadata, partitionBy: partCol, partitions: String(groups.size) },
      };
    }

    return this.executeAppendSingle(table, rows, options);
  }

  /** Write a single fragment (no partitioning). */
  private async executeAppendSingle(table: string, rows: Record<string, unknown>[], options?: AppendOptions): Promise<AppendResult> {

    const wasm = await this.getWasm();

    // Convert row-major to column-major
    const columnArrays = rowsToColumnArrays(rows);
    if (columnArrays.length === 0) throw new Error("No valid columns found");

    // Build Lance fragment via WASM
    const fragmentBytes = wasm.buildFragment(columnArrays);

    // Generate unique data file path — options.path overrides default location
    const uuid = crypto.randomUUID();
    const r2Prefix = options?.path
      ? (options.path.endsWith("/") ? options.path : `${options.path}/`)
      : (table.endsWith(".lance/") || table.endsWith("/") ? table : `${table}.lance/`);
    const dataFilePath = `data/${uuid}.lance`;
    const dataR2Key = `${r2Prefix}${dataFilePath}`;

    // PUT data file to R2 (unique name = no conflict)
    await resolveBucket(this.env, dataR2Key).put(dataR2Key, fragmentBytes);

    // CAS loop for manifest update
    const MAX_RETRIES = 10;
    const latestKey = `${r2Prefix}_versions/_latest`;

    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      // Read current _latest with ETag
      const latestObj = await resolveBucket(this.env, latestKey).get(latestKey);
      let currentVersion = 0;
      let etag: string | undefined;

      if (latestObj) {
        const text = await latestObj.text();
        currentVersion = parseInt(text.trim(), 10) || 0;
        etag = latestObj.etag;
      }

      const newVersion = currentVersion + 1;
      const newVersionStr = String(newVersion);

      // Build manifest: simple text format with fragment list
      // Read existing manifest if available
      let existingFragments: { id: number; filePath: string; physicalRows: number }[] = [];
      if (currentVersion > 0) {
        const manifestKey = `${r2Prefix}_versions/${currentVersion}.manifest`;
        const manifestObj = await resolveBucket(this.env, manifestKey).get(manifestKey);
        if (manifestObj) {
          const manifest = parseManifest(await manifestObj.arrayBuffer());
          if (manifest) existingFragments = manifest.fragments;
        }
      }

      // Add new fragment
      const newFragments = [
        ...existingFragments,
        { id: existingFragments.length + 1, filePath: dataFilePath, physicalRows: rows.length },
      ];

      // Write new manifest (protobuf-compatible binary)
      const manifestPayload = this.buildManifestBinary(newVersion, newFragments);
      const newManifestKey = `${r2Prefix}_versions/${newVersion}.manifest`;
      await resolveBucket(this.env, newManifestKey).put(newManifestKey, manifestPayload);

      // CAS: write _latest with ETag condition
      try {
        const putOptions: R2PutOptions = etag ? { onlyIf: { etagMatches: etag } } : {};
        const result = await resolveBucket(this.env, latestKey).put(latestKey, newVersionStr, putOptions);

        if (result === null && etag) {
          // 412 Precondition Failed — retry
          continue;
        }

        // Success — broadcast invalidation
        const tableName = table.replace(/\.lance\/?$/, "").split("/").pop() ?? table;
        const totalRows = newFragments.reduce((s, f) => s + f.physicalRows, 0);
        const footerResult = await this.readFooterAndColumns(dataR2Key);
        if (footerResult) {
          await this.broadcast(tableName, dataR2Key, footerResult, { totalRows });
        }

        // Update stored table meta (include metadata if provided)
        const tableMeta: Record<string, unknown> = {
          name: tableName, r2Prefix, totalRows, updatedAt: Date.now(),
        };
        if (options?.metadata) tableMeta.writeMetadata = options.metadata;
        await this.ctx.storage.put(`table:${tableName}`, tableMeta);

        return {
          version: newVersion,
          dataFilePath,
          retries: attempt,
          rowsWritten: rows.length,
          ...(options?.metadata ? { metadata: options.metadata } : {}),
        } satisfies AppendResult;
      } catch {
        // R2 conditional put failure — retry
        continue;
      }
    }

    throw new Error("CAS failed after max retries");
  }

  /** Build a simple binary manifest for the _versions/ directory. */
  private buildManifestBinary(
    version: number,
    fragments: { id: number; filePath: string; physicalRows: number }[],
  ): ArrayBuffer {
    // Simple protobuf-like format matching parseManifest expectations
    // Version field (tag 1, varint): version number
    // Fragment fields (tag 2, length-delimited): each fragment
    const enc = textEncoder;
    const parts: Uint8Array[] = [];

    // Write version varint (field 1)
    parts.push(new Uint8Array([0x08])); // tag 1, wire type 0
    parts.push(this.encodeVarint(version));

    // Write each fragment
    for (const frag of fragments) {
      const pathBytes = enc.encode(frag.filePath);
      const fragParts: Uint8Array[] = [];

      // Fragment ID (field 1, varint)
      fragParts.push(new Uint8Array([0x08]));
      fragParts.push(this.encodeVarint(frag.id));

      // File path (field 2, length-delimited)
      fragParts.push(new Uint8Array([0x12]));
      fragParts.push(this.encodeVarint(pathBytes.length));
      fragParts.push(pathBytes);

      // Physical rows (field 3, varint)
      fragParts.push(new Uint8Array([0x18]));
      fragParts.push(this.encodeVarint(frag.physicalRows));

      // Combine fragment
      let fragLen = 0;
      for (const p of fragParts) fragLen += p.length;
      const fragBuf = new Uint8Array(fragLen);
      let off = 0;
      for (const p of fragParts) { fragBuf.set(p, off); off += p.length; }

      // Fragment as field 2, length-delimited
      parts.push(new Uint8Array([0x12]));
      parts.push(this.encodeVarint(fragLen));
      parts.push(fragBuf);
    }

    let totalLen = 0;
    for (const p of parts) totalLen += p.length;
    const result = new Uint8Array(totalLen);
    let off = 0;
    for (const p of parts) { result.set(p, off); off += p.length; }

    return result.buffer;
  }

  private encodeVarint(value: number): Uint8Array {
    const bytes: number[] = [];
    let v = value >>> 0; // ensure unsigned
    while (v > 0x7f) {
      bytes.push((v & 0x7f) | 0x80);
      v >>>= 7;
    }
    bytes.push(v & 0x7f);
    return new Uint8Array(bytes);
  }

  async refreshRpc(body: unknown): Promise<unknown> {
    const { r2Key } = body as { r2Key: string };
    if (!r2Key || typeof r2Key !== "string" || r2Key.includes("..")) {
      throw new Error("Invalid r2Key");
    }
    const result = await this.readFooterAndColumns(r2Key);
    if (!result) throw new Error("Failed to read footer");

    const tableName = r2Key.replace(/\.(lance|parquet)$/, "").split("/").pop() ?? r2Key;
    await this.broadcast(tableName, r2Key, result);
    return { refreshed: true, table: tableName };
  }

  async listTablesRpc(): Promise<{ tables: string[] }> {
    const tables = await this.ctx.storage.list<TableMeta>({ prefix: "table:" });
    return { tables: [...tables.keys()].map(k => k.replace("table:", "")) };
  }

  /** Read footer + column metadata from R2 (2 range reads, done once by Master). */
  private async readFooterAndColumns(r2Key: string): Promise<{
    parsed?: Footer; raw: ArrayBuffer; fileSize: bigint; columns: ColumnMeta[];
    format: "lance" | "parquet";
  } | null> {
    const head = await withTimeout(resolveBucket(this.env, r2Key).head(r2Key), 10_000);
    if (!head) return null;

    const fileSize = BigInt(head.size);
    // Read last 40 bytes — enough for Lance footer or Parquet tail detection
    const tailSize = Math.min(Number(fileSize), FOOTER_SIZE);
    const obj = await withTimeout(resolveBucket(this.env, r2Key).get(r2Key, {
      range: { offset: Number(fileSize) - tailSize, length: tailSize },
    }), 10_000);
    if (!obj) return null;

    const raw = await obj.arrayBuffer();
    const format = detectFormat(raw);

    if (format === "parquet") {
      const footerLen = getParquetFooterLength(raw);
      if (!footerLen) return null;

      // Fetch full Parquet Thrift footer
      const footerOffset = Number(fileSize) - footerLen - 8;
      const footerObj = await withTimeout(resolveBucket(this.env, r2Key).get(r2Key, {
        range: { offset: footerOffset, length: footerLen },
      }), 10_000);
      if (!footerObj) return null;

      const footerBuf = await footerObj.arrayBuffer();
      const parquetMeta = parseParquetFooter(footerBuf);
      if (!parquetMeta) return null;

      const tableMeta = parquetMetaToTableMeta(parquetMeta, r2Key, fileSize);
      return { raw, fileSize, columns: tableMeta.columns, format: "parquet" };
    }

    // Lance format
    const parsed = parseFooter(raw);
    if (!parsed) return null;

    let columns: ColumnMeta[] = [];
    const metaLen = Number(parsed.columnMetaOffsetsStart) - Number(parsed.columnMetaStart);
    if (metaLen > 0) {
      const metaObj = await withTimeout(resolveBucket(this.env, r2Key).get(r2Key, {
        range: { offset: Number(parsed.columnMetaStart), length: metaLen },
      }), 10_000);
      if (metaObj) columns = parseColumnMetaFromProtobuf(await metaObj.arrayBuffer(), parsed.numColumns);
    }

    return { parsed, raw, fileSize, columns, format: "lance" };
  }

  /** Broadcast invalidation with pre-parsed columns to all Query DOs via RPC. */
  private async broadcast(
    table: string, r2Key: string,
    footer: { raw: ArrayBuffer; fileSize: bigint; columns: ColumnMeta[]; format?: "lance" | "parquet" },
    opts?: { totalRows?: number; r2Prefix?: string },
  ): Promise<void> {
    const regions = (await this.ctx.storage.get<Record<string, string>>("regions")) ?? {};
    const payload = {
      table, r2Key, columns: footer.columns, format: footer.format ?? "lance",
      footerRaw: footer.raw,
      fileSize: footer.fileSize, timestamp: Date.now(),
      ...(opts?.totalRows != null ? { totalRows: opts.totalRows } : {}),
      ...(opts?.r2Prefix != null ? { r2Prefix: opts.r2Prefix } : {}),
    };

    const deadRegions: string[] = [];
    await Promise.allSettled(Object.entries(regions).map(async ([region, doId]) => {
      try {
        const queryDo = this.env.QUERY_DO.get(this.env.QUERY_DO.idFromString(doId)) as unknown as QueryDORpc;
        await withTimeout(queryDo.invalidateRpc(payload), 5_000);
        this.broadcastFailures.delete(region);
      } catch {
        const count = (this.broadcastFailures.get(region) ?? 0) + 1;
        this.broadcastFailures.set(region, count);
        if (count >= 3) {
          console.warn(`Broadcast to ${region} failed ${count} times, removing`);
          deadRegions.push(region);
        } else {
          console.warn(`Broadcast to ${region} failed (${count}/3)`);
        }
      }
    }));

    if (deadRegions.length > 0) {
      for (const r of deadRegions) {
        delete regions[r];
        this.broadcastFailures.delete(r);
      }
      await this.ctx.storage.put("regions", regions);
    }
  }

  /** RPC: Append rows — zero-serialization call from RemoteExecutor. */
  async appendRpc(table: string, rows: Record<string, unknown>[], options?: AppendOptions): Promise<AppendResult> {
    return this.executeAppend(table, rows, options);
  }

  /** RPC: Drop a table — delete all R2 objects and DO metadata. */
  async dropRpc(table: string): Promise<DropResult> {
    const tableName = table.replace(/\.lance\/?$/, "").split("/").pop() ?? table;
    const meta = await this.ctx.storage.get<Record<string, unknown>>(`table:${tableName}`);

    let fragmentsDeleted = 0;
    let bytesFreed = 0;

    if (meta) {
      const r2Prefix = (meta.r2Prefix as string) ?? `${tableName}.lance/`;

      // List and delete all R2 objects under this prefix
      let cursor: string | undefined;
      do {
        const listed = await resolveBucket(this.env, r2Prefix).list({
          prefix: r2Prefix,
          cursor,
          limit: 1000,
        });

        if (listed.objects.length > 0) {
          const keys = listed.objects.map(o => o.key);
          bytesFreed += listed.objects.reduce((s, o) => s + o.size, 0);
          fragmentsDeleted += keys.length;
          // R2 delete supports up to 1000 keys per call
          await resolveBucket(this.env, r2Prefix).delete(keys);
        }

        cursor = listed.truncated ? listed.cursor : undefined;
      } while (cursor);

      // Remove DO metadata
      await this.ctx.storage.delete(`table:${tableName}`);

      // Broadcast invalidation (empty footer signals deletion)
      const regions = (await this.ctx.storage.get<Record<string, string>>("regions")) ?? {};
      await Promise.allSettled(Object.entries(regions).map(async ([, doId]) => {
        try {
          const queryDo = this.env.QUERY_DO.get(this.env.QUERY_DO.idFromString(doId)) as unknown as QueryDORpc;
          await queryDo.invalidateRpc({ table: tableName, deleted: true, timestamp: Date.now() });
        } catch { /* best-effort broadcast */ }
      }));
    }

    return { table: tableName, fragmentsDeleted, bytesFreed };
  }
}
