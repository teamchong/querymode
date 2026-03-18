import type { Row } from "./types.js";
import { NULL_SENTINEL } from "./types.js";
import type { QueryDescriptor } from "./client.js";

export interface MaterializationCacheOptions {
  /** Maximum bytes for the cache (default 64MB). */
  maxBytes?: number;
}

interface CacheEntry {
  rows: Row[];
  byteSize: number;
  lastAccess: number;
}

const DEFAULT_MAX_BYTES = 64 * 1024 * 1024; // 64MB

/**
 * Estimate the byte size of a single Row.
 * 64 bytes base overhead + 8 bytes per key + value sizes.
 */
function estimateRowBytes(row: Row): number {
  let bytes = 64; // base object overhead
  for (const key in row) {
    bytes += 8; // key pointer/overhead
    const val = row[key];
    if (val === null || val === undefined) {
      bytes += 0;
    } else if (typeof val === "number") {
      bytes += 8;
    } else if (typeof val === "bigint") {
      bytes += 8;
    } else if (typeof val === "boolean") {
      bytes += 4;
    } else if (typeof val === "string") {
      bytes += val.length * 2;
    } else if (val instanceof Float32Array) {
      bytes += val.byteLength;
    }
  }
  return bytes;
}

/**
 * Estimate total byte size for a page of rows.
 */
function estimatePageBytes(rows: Row[]): number {
  let total = 0;
  for (let i = 0; i < rows.length; i++) {
    total += estimateRowBytes(rows[i]);
  }
  return total;
}

/**
 * Page-level LRU cache for materialized query results.
 *
 * Keyed by query hash + page offset. Evicts least-recently-accessed
 * entries when the byte budget is exceeded.
 */
export class MaterializationCache {
  private readonly maxBytes: number;
  private readonly entries: Map<string, CacheEntry> = new Map();
  private _bytesUsed = 0;
  private _hits = 0;
  private _misses = 0;

  constructor(opts?: MaterializationCacheOptions) {
    this.maxBytes = opts?.maxBytes ?? DEFAULT_MAX_BYTES;
  }

  /** Store a page of rows under a query hash + offset. */
  set(queryHash: string, offset: number, rows: Row[]): void {
    const key = `${queryHash}:${offset}`;
    // Remove existing entry if present (to update size accounting)
    const existing = this.entries.get(key);
    if (existing) {
      this._bytesUsed -= existing.byteSize;
      this.entries.delete(key);
    }

    const byteSize = estimatePageBytes(rows);

    // If single entry exceeds budget, don't cache it
    if (byteSize > this.maxBytes) return;

    // Evict LRU entries until there's room
    while (this._bytesUsed + byteSize > this.maxBytes && this.entries.size > 0) {
      this.evictLRU();
    }

    this.entries.set(key, {
      rows,
      byteSize,
      lastAccess: performance.now(),
    });
    this._bytesUsed += byteSize;
  }

  /**
   * Retrieve a cached page. Returns null on miss.
   *
   * A get for offset=100, limit=50 succeeds if there is a cached page
   * starting at offset=100 that contains at least 50 rows.
   */
  get(queryHash: string, offset: number, limit: number): Row[] | null {
    const key = `${queryHash}:${offset}`;
    const entry = this.entries.get(key);
    if (!entry || entry.rows.length < limit) {
      this._misses++;
      return null;
    }
    // Update last access for LRU
    entry.lastAccess = performance.now();
    this._hits++;
    // Return exactly `limit` rows if the cached page has more
    return entry.rows.length === limit ? entry.rows : entry.rows.slice(0, limit);
  }

  /** Invalidate all cached pages for a query. */
  invalidate(queryHash: string): void {
    const prefix = `${queryHash}:`;
    for (const [key, entry] of this.entries) {
      if (key.startsWith(prefix)) {
        this._bytesUsed -= entry.byteSize;
        this.entries.delete(key);
      }
    }
  }

  /** Clear the entire cache. */
  clear(): void {
    this.entries.clear();
    this._bytesUsed = 0;
  }

  /** Cache statistics. */
  get stats(): { hits: number; misses: number; bytesUsed: number; entries: number } {
    return {
      hits: this._hits,
      misses: this._misses,
      bytesUsed: this._bytesUsed,
      entries: this.entries.size,
    };
  }

  /** Evict the least-recently-accessed entry. */
  private evictLRU(): void {
    let oldestKey: string | undefined;
    let oldestTime = Infinity;
    for (const [key, entry] of this.entries) {
      if (entry.lastAccess < oldestTime) {
        oldestTime = entry.lastAccess;
        oldestKey = key;
      }
    }
    if (oldestKey !== undefined) {
      const entry = this.entries.get(oldestKey)!;
      this._bytesUsed -= entry.byteSize;
      this.entries.delete(oldestKey);
    }
  }
}

/**
 * Create a deterministic hash key from a QueryDescriptor, excluding
 * offset and limit (since those define the page, not the query identity).
 */
export function queryHashKey(desc: QueryDescriptor): string {
  const parts: string[] = [desc.table];

  // Filters — sorted by column for determinism
  if (desc.filters.length > 0) {
    const sorted = [...desc.filters].sort((a, b) => a.column.localeCompare(b.column));
    for (const f of sorted) {
      parts.push(`f:${f.column}:${f.op}:${stringifyValue(f.value)}`);
    }
  }

  // Projections — sorted
  if (desc.projections.length > 0) {
    parts.push(`p:${[...desc.projections].sort().join(",")}`);
  }

  // Sort
  if (desc.sortColumn) {
    parts.push(`s:${desc.sortColumn}:${desc.sortDirection ?? "asc"}`);
  }

  // Aggregates
  if (desc.aggregates && desc.aggregates.length > 0) {
    for (const agg of desc.aggregates) {
      parts.push(`a:${agg.fn}:${agg.column}${agg.alias ? `:${agg.alias}` : ""}`);
    }
  }

  // Group by
  if (desc.groupBy && desc.groupBy.length > 0) {
    parts.push(`g:${[...desc.groupBy].sort().join(",")}`);
  }

  // Filter groups (OR)
  if (desc.filterGroups && desc.filterGroups.length > 0) {
    for (let gi = 0; gi < desc.filterGroups.length; gi++) {
      const grp = desc.filterGroups[gi];
      for (const f of grp) parts.push(`fg${gi}:${f.column}:${f.op}:${stringifyValue(f.value)}`);
    }
  }

  // Distinct
  if (desc.distinct && desc.distinct.length > 0) {
    parts.push(`d:${[...desc.distinct].sort().join(",")}`);
  }

  // Vector search
  if (desc.vectorSearch) {
    const vs = desc.vectorSearch;
    let vecStr = "";
    for (let i = 0; i < vs.queryVector.length; i++) { if (i > 0) vecStr += ","; vecStr += vs.queryVector[i]; }
    parts.push(`v:${vs.column}:${vs.topK}:${vecStr}`);
  }

  // Windows
  if (desc.windows && desc.windows.length > 0) {
    for (const w of desc.windows) {
      let wp = `w:${w.fn}:${w.alias}:${w.column ?? NULL_SENTINEL}`;
      if (w.partitionBy?.length) wp += `:pb=${w.partitionBy.join(",")}`;
      if (w.orderBy?.length) wp += `:ob=${w.orderBy.map(o => o.column + o.direction).join(",")}`;
      if (w.frame) wp += `:fr=${w.frame.type}:${w.frame.start}:${w.frame.end}`;
      if (w.args?.offset !== undefined) wp += `:ao=${w.args.offset}`;
      if (w.args?.default_ !== undefined) wp += `:ad=${w.args.default_}`;
      parts.push(wp);
    }
  }

  // Computed columns
  if (desc.computedColumns && desc.computedColumns.length > 0) {
    for (const cc of desc.computedColumns) parts.push(`cc:${cc.alias}`);
  }

  // Set operation
  if (desc.setOperation) {
    parts.push(`so:${desc.setOperation.mode}:${queryHashKey(desc.setOperation.right as QueryDescriptor)}`);
  }

  // Subquery IN
  if (desc.subqueryIn) {
    for (const sq of desc.subqueryIn) {
      parts.push(`sq:${sq.column}:${[...sq.valueSet].sort().join(",")}`);
    }
  }

  // Join
  if (desc.join) {
    const j = desc.join;
    parts.push(`j:${j.leftKey}:${j.rightKey}:${j.type ?? "inner"}:${queryHashKey(j.right as QueryDescriptor)}`);
  }

  // Version
  if (desc.version !== undefined) {
    parts.push(`ver:${desc.version}`);
  }

  return parts.join("|");
}

function stringifyValue(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(stringifyValue).join(",")}]`;
  }
  if (typeof value === "bigint") {
    return `${value}n`;
  }
  return String(value);
}
