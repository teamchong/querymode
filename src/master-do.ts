import type { ColumnMeta, Env, Footer, TableMeta, DatasetMeta, AppendResult } from "./types.js";
import { parseFooter, parseColumnMetaFromProtobuf, FOOTER_SIZE } from "./footer.js";
import { parseManifest } from "./manifest.js";
import { detectFormat, getParquetFooterLength, parseParquetFooter, parquetMetaToTableMeta } from "./parquet.js";
import { instantiateWasm, type WasmEngine } from "./wasm-engine.js";

/** Master DO — single writer, reads footers, broadcasts invalidations. */
export class MasterDO implements DurableObject {
  private state: DurableObjectState;
  private env: Env;
  private broadcastFailures = new Map<string, number>(); // region → consecutive failure count
  private wasmEngine?: WasmEngine;

  constructor(state: DurableObjectState, env: Env) {
    this.state = state;
    this.env = env;
  }

  private json(body: unknown, status = 200): Response {
    return new Response(JSON.stringify(body), {
      status, headers: { "content-type": "application/json" },
    });
  }

  async fetch(request: Request): Promise<Response> {
    switch (new URL(request.url).pathname) {
      case "/register": return this.handleRegister(request);
      case "/write":    return this.handleWrite(request);
      case "/append":   return this.handleAppend(request);
      case "/refresh":  return this.handleRefresh(request);
      case "/tables":   return this.handleListTables();
      default:          return new Response("Not found", { status: 404 });
    }
  }

  private async handleRegister(request: Request): Promise<Response> {
    const { queryDoId, region } = (await request.json()) as { queryDoId: string; region: string };
    const regions = (await this.state.storage.get<Record<string, string>>("regions")) ?? {};
    regions[region] = queryDoId;
    await this.state.storage.put("regions", regions);

    // Return current table timestamps so waking Query DOs can detect stale caches
    const tables = await this.state.storage.list<TableMeta>({ prefix: "table:" });
    const tableVersions: Record<string, { r2Key: string; updatedAt: number }> = {};
    for (const [key, meta] of tables) {
      const name = key.replace("table:", "");
      tableVersions[name] = { r2Key: meta.r2Key ?? name, updatedAt: meta.updatedAt ?? 0 };
    }
    return this.json({ registered: true, region, tableVersions });
  }

  private async handleWrite(request: Request): Promise<Response> {
    const { r2Key } = (await request.json()) as { r2Key: string };

    // Check if this is a dataset directory (ends with / or .lance/)
    if (r2Key.endsWith("/") || r2Key.endsWith(".lance/")) {
      return this.handleDatasetWrite(r2Key);
    }

    const result = await this.readFooterAndColumns(r2Key);
    if (!result) return this.json({ error: "Failed to read footer" }, 500);

    const tableName = r2Key.replace(/\.(lance|parquet)$/, "").split("/").pop() ?? r2Key;
    const meta: TableMeta = {
      name: tableName, footer: result.parsed, format: result.format, columns: result.columns,
      totalRows: result.columns[0]?.pages.reduce((s, p) => s + p.rowCount, 0) ?? 0,
      fileSize: result.fileSize, r2Key, updatedAt: Date.now(),
    };
    await this.state.storage.put(`table:${tableName}`, meta);
    await this.broadcast(tableName, r2Key, result);
    return this.json({ success: true, table: tableName });
  }

  /** Handle write notification for a multi-fragment dataset directory. */
  private async handleDatasetWrite(r2Prefix: string): Promise<Response> {
    const tableName = r2Prefix.replace(/\/$/, "").replace(/\.lance$/, "").split("/").pop() ?? r2Prefix;

    // Find latest manifest
    const listed = await this.env.DATA_BUCKET.list({ prefix: `${r2Prefix}_versions/`, limit: 100 });
    const manifestKeys = listed.objects
      .filter(o => o.key.endsWith(".manifest"))
      .sort((a, b) => a.key.localeCompare(b.key));
    if (manifestKeys.length === 0) return this.json({ error: "No manifests found" }, 404);

    const latestKey = manifestKeys[manifestKeys.length - 1].key;
    const manifestObj = await this.env.DATA_BUCKET.get(latestKey);
    if (!manifestObj) return this.json({ error: "Failed to read manifest" }, 500);

    const manifest = parseManifest(await manifestObj.arrayBuffer());
    if (!manifest) return this.json({ error: "Failed to parse manifest" }, 500);

    // Read first fragment's footer to broadcast (Query DOs will discover the rest)
    if (manifest.fragments.length > 0) {
      const firstFrag = manifest.fragments[0];
      const fragKey = `${r2Prefix}${firstFrag.filePath}`;
      const result = await this.readFooterAndColumns(fragKey);
      if (result) {
        await this.broadcast(tableName, fragKey, result);
      }
    }

    await this.state.storage.put(`table:${tableName}`, {
      name: tableName, r2Prefix, manifest, totalRows: manifest.totalRows, updatedAt: Date.now(),
    });

    return this.json({ success: true, table: tableName, fragments: manifest.fragments.length, totalRows: manifest.totalRows });
  }

  private async getWasm(): Promise<WasmEngine> {
    if (this.wasmEngine) return this.wasmEngine;
    this.wasmEngine = await instantiateWasm(this.env.EDGEQ_WASM);
    return this.wasmEngine;
  }

  /** Append rows to a table using CAS coordination.
   *  1. Build Lance fragment from row data via WASM
   *  2. PUT data file to R2 (unique name, no conflict)
   *  3. CAS loop: read _latest → build new manifest → PUT with ETag match
   */
  private async handleAppend(request: Request): Promise<Response> {
    const { table, rows } = (await request.json()) as {
      table: string;
      rows: Record<string, unknown>[];
    };

    if (!rows?.length) return this.json({ error: "No rows provided" }, 400);

    const wasm = await this.getWasm();

    // Convert row-major to column-major
    const columnNames = Object.keys(rows[0]);
    const columnArrays: { name: string; dtype: string; values: ArrayBufferLike }[] = [];

    for (const colName of columnNames) {
      const sampleValue = rows.find(r => r[colName] != null)?.[colName];
      if (sampleValue === undefined) continue;

      if (typeof sampleValue === "number") {
        if (Number.isInteger(sampleValue)) {
          const i64 = new BigInt64Array(rows.length);
          for (let i = 0; i < rows.length; i++) i64[i] = BigInt(rows[i][colName] as number);
          columnArrays.push({ name: colName, dtype: "int64", values: i64.buffer });
        } else {
          const f64 = new Float64Array(rows.length);
          for (let i = 0; i < rows.length; i++) f64[i] = rows[i][colName] as number;
          columnArrays.push({ name: colName, dtype: "float64", values: f64.buffer });
        }
      } else if (typeof sampleValue === "bigint") {
        const i64 = new BigInt64Array(rows.length);
        for (let i = 0; i < rows.length; i++) i64[i] = rows[i][colName] as bigint;
        columnArrays.push({ name: colName, dtype: "int64", values: i64.buffer });
      } else if (typeof sampleValue === "string") {
        // Length-prefixed encoding
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
        for (const part of parts) { buf.set(part, off); off += part.length; }
        columnArrays.push({ name: colName, dtype: "utf8", values: buf.buffer });
      } else if (typeof sampleValue === "boolean") {
        const byteCount = Math.ceil(rows.length / 8);
        const boolBuf = new Uint8Array(byteCount);
        for (let i = 0; i < rows.length; i++) {
          if (rows[i][colName]) boolBuf[i >> 3] |= 1 << (i & 7);
        }
        columnArrays.push({ name: colName, dtype: "bool", values: boolBuf.buffer });
      }
    }

    if (columnArrays.length === 0) return this.json({ error: "No valid columns found" }, 400);

    // Build Lance fragment via WASM
    const fragmentBytes = wasm.buildFragment(columnArrays);

    // Generate unique data file path
    const uuid = crypto.randomUUID();
    const r2Prefix = table.endsWith(".lance/") || table.endsWith("/") ? table : `${table}.lance/`;
    const dataFilePath = `data/${uuid}.lance`;
    const dataR2Key = `${r2Prefix}${dataFilePath}`;

    // PUT data file to R2 (unique name = no conflict)
    await this.env.DATA_BUCKET.put(dataR2Key, fragmentBytes);

    // CAS loop for manifest update
    const MAX_RETRIES = 10;
    const latestKey = `${r2Prefix}_versions/_latest`;

    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      // Read current _latest with ETag
      const latestObj = await this.env.DATA_BUCKET.get(latestKey);
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
        const manifestObj = await this.env.DATA_BUCKET.get(manifestKey);
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
      await this.env.DATA_BUCKET.put(newManifestKey, manifestPayload);

      // CAS: write _latest with ETag condition
      try {
        const putOptions: R2PutOptions = etag ? { onlyIf: { etagMatches: etag } } : {};
        const result = await this.env.DATA_BUCKET.put(latestKey, newVersionStr, putOptions);

        if (result === null && etag) {
          // 412 Precondition Failed — retry
          continue;
        }

        // Success — broadcast invalidation
        const footerResult = await this.readFooterAndColumns(dataR2Key);
        if (footerResult) {
          await this.broadcast(table.replace(/\.lance\/?$/, "").split("/").pop() ?? table, dataR2Key, footerResult);
        }

        // Update stored table meta
        const tableName = table.replace(/\.lance\/?$/, "").split("/").pop() ?? table;
        const totalRows = newFragments.reduce((s, f) => s + f.physicalRows, 0);
        await this.state.storage.put(`table:${tableName}`, {
          name: tableName, r2Prefix, totalRows, updatedAt: Date.now(),
        });

        return this.json({
          version: newVersion,
          dataFilePath,
          retries: attempt,
          rowsWritten: rows.length,
        } satisfies AppendResult);
      } catch {
        // R2 conditional put failure — retry
        continue;
      }
    }

    return this.json({ error: "CAS failed after max retries" }, 409);
  }

  /** Build a simple binary manifest for the _versions/ directory. */
  private buildManifestBinary(
    version: number,
    fragments: { id: number; filePath: string; physicalRows: number }[],
  ): ArrayBuffer {
    // Simple protobuf-like format matching parseManifest expectations
    // Version field (tag 1, varint): version number
    // Fragment fields (tag 2, length-delimited): each fragment
    const enc = new TextEncoder();
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

  private async handleRefresh(request: Request): Promise<Response> {
    const { r2Key } = (await request.json()) as { r2Key: string };
    const result = await this.readFooterAndColumns(r2Key);
    if (!result) return this.json({ error: "Failed to read footer" }, 500);

    const tableName = r2Key.replace(/\.lance$/, "").split("/").pop() ?? r2Key;
    await this.broadcast(tableName, r2Key, result);
    return this.json({ refreshed: true, table: tableName });
  }

  private async handleListTables(): Promise<Response> {
    const tables = await this.state.storage.list<TableMeta>({ prefix: "table:" });
    return this.json({ tables: [...tables.keys()].map(k => k.replace("table:", "")) });
  }

  /** Read footer + column metadata from R2 (2 range reads, done once by Master). */
  private async readFooterAndColumns(r2Key: string): Promise<{
    parsed?: Footer; raw: ArrayBuffer; fileSize: bigint; columns: ColumnMeta[];
    format: "lance" | "parquet";
  } | null> {
    const head = await this.env.DATA_BUCKET.head(r2Key);
    if (!head) return null;

    const fileSize = BigInt(head.size);
    // Read last 40 bytes — enough for Lance footer or Parquet tail detection
    const tailSize = Math.min(Number(fileSize), FOOTER_SIZE);
    const obj = await this.env.DATA_BUCKET.get(r2Key, {
      range: { offset: Number(fileSize) - tailSize, length: tailSize },
    });
    if (!obj) return null;

    const raw = await obj.arrayBuffer();
    const format = detectFormat(raw);

    if (format === "parquet") {
      const footerLen = getParquetFooterLength(raw);
      if (!footerLen) return null;

      // Fetch full Parquet Thrift footer
      const footerOffset = Number(fileSize) - footerLen - 8;
      const footerObj = await this.env.DATA_BUCKET.get(r2Key, {
        range: { offset: footerOffset, length: footerLen },
      });
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
      const metaObj = await this.env.DATA_BUCKET.get(r2Key, {
        range: { offset: Number(parsed.columnMetaStart), length: metaLen },
      });
      if (metaObj) columns = parseColumnMetaFromProtobuf(await metaObj.arrayBuffer(), parsed.numColumns);
    }

    return { parsed, raw, fileSize, columns, format: "lance" };
  }

  /** Broadcast invalidation with pre-parsed columns to all Query DOs. */
  private async broadcast(
    table: string, r2Key: string,
    footer: { raw: ArrayBuffer; fileSize: bigint; columns: ColumnMeta[]; format?: "lance" | "parquet" },
  ): Promise<void> {
    const regions = (await this.state.storage.get<Record<string, string>>("regions")) ?? {};
    const payload = JSON.stringify({
      table, r2Key, columns: footer.columns, format: footer.format ?? "lance",
      footerBytes: Array.from(new Uint8Array(footer.raw)),
      fileSize: footer.fileSize.toString(), timestamp: Date.now(),
    });

    const deadRegions: string[] = [];
    await Promise.allSettled(Object.entries(regions).map(async ([region, doId]) => {
      try {
        const queryDo = this.env.QUERY_DO.get(this.env.QUERY_DO.idFromString(doId));
        await queryDo.fetch(new Request("http://internal/invalidate", {
          method: "POST", body: payload, headers: { "content-type": "application/json" },
        }));
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
      await this.state.storage.put("regions", regions);
    }
  }
}
