/**
 * PartitionCatalog — lightweight index mapping partition key values to fragment IDs.
 *
 * At PB scale with millions of fragments, evaluating min/max stats on every fragment
 * is too slow. The partition catalog provides O(1) lookup: given filter values,
 * return only the fragment IDs that could contain matching rows.
 *
 * Built during ingest or on first full scan, cached in DO memory (~1KB per partition value).
 *
 * Example: if data is partitioned by `region`, the catalog maps:
 *   "us" → [frag-1, frag-5, frag-12]
 *   "eu" → [frag-2, frag-8]
 *   "asia" → [frag-3, frag-9, frag-15]
 *
 * A query with `WHERE region = 'us'` skips to [frag-1, frag-5, frag-12] directly.
 */

import type { FilterOp, TableMeta } from "./types.js";
import { NULL_SENTINEL } from "./types.js";

/** Convert a partition value to a consistent string key, using NULL_SENTINEL for null/undefined. */
function partKey(v: unknown): string {
  return v === null || v === undefined ? NULL_SENTINEL : String(v);
}

/** A partition value mapped to the fragment IDs that contain it. */
export interface PartitionEntry {
  value: string | number | bigint;
  fragmentIds: number[];
}

/** In-memory partition catalog for a single table. */
export class PartitionCatalog {
  /** Column name this catalog is partitioned on. */
  readonly column: string;
  /** Map of stringified value → fragment IDs. */
  private index = new Map<string, number[]>();
  /** All fragment IDs in the catalog (for filters that can't use the index). */
  private allFragmentIds: number[] = [];
  /** O(1) dedup during construction/registration. */
  private allFragmentIdSet = new Set<number>();
  /** O(1) per-key dedup for register() — avoids O(n) includes() scan on hot ingest path. */
  private _indexSets = new Map<string, Set<number>>();
  /** True when every indexed page has min===max (Hive-style partitioning).
   *  Only exact-partition catalogs are safe for eq/in lookups. */
  private exactPartition = true;

  constructor(column: string) {
    this.column = column;
  }

  /** Build catalog from fragment metadata by extracting min/max per fragment. */
  static fromFragments(column: string, fragments: Map<number, TableMeta>): PartitionCatalog {
    const catalog = new PartitionCatalog(column);
    // Use Sets during construction for O(1) dedup
    const indexSets = new Map<string, Set<number>>();

    for (const [fragId, meta] of fragments) {
      catalog.allFragmentIdSet.add(fragId);
      const col = meta.columns.find(c => c.name === column);
      if (!col) continue;

      const values = new Set<string>();
      for (const page of col.pages) {
        if (page.minValue !== undefined) values.add(partKey(page.minValue));
        if (page.maxValue !== undefined) values.add(partKey(page.maxValue));
        // If min !== max on any page, this isn't exact-partition data
        if (page.minValue !== undefined && page.maxValue !== undefined &&
            partKey(page.minValue) !== partKey(page.maxValue)) {
          catalog.exactPartition = false;
        }
      }

      for (const v of values) {
        let entry = indexSets.get(v);
        if (!entry) { entry = new Set(); indexSets.set(v, entry); }
        entry.add(fragId);
      }
    }

    catalog.allFragmentIds = [...catalog.allFragmentIdSet];
    for (const [key, set] of indexSets) {
      catalog.index.set(key, [...set]);
    }
    return catalog;
  }

  /** Register a fragment's partition values (used during ingest). */
  register(fragmentId: number, values: (string | number | bigint)[]): void {
    if (!this.allFragmentIdSet.has(fragmentId)) {
      this.allFragmentIdSet.add(fragmentId);
      this.allFragmentIds.push(fragmentId);
    }
    for (const v of values) {
      const key = partKey(v);
      let setEntry = this._indexSets.get(key);
      if (!setEntry) { setEntry = new Set(); this._indexSets.set(key, setEntry); }
      if (!setEntry.has(fragmentId)) {
        setEntry.add(fragmentId);
        let entry = this.index.get(key);
        if (!entry) { entry = []; this.index.set(key, entry); }
        entry.push(fragmentId);
      }
    }
  }

  /**
   * Prune fragment IDs using filters on the partition column.
   * Returns only fragment IDs that could match the filter.
   * Returns null if the filter can't be pruned (use all fragments).
   */
  prune(filters: FilterOp[]): number[] | null {
    const relevant = filters.filter(f => f.column === this.column);
    if (relevant.length === 0) return null; // no partition filter → can't prune

    let result: Set<number> | null = null;

    for (const filter of relevant) {
      const matching = this.matchFilter(filter);
      if (!matching) return null; // can't evaluate this filter → skip catalog

      if (result === null) {
        result = new Set(matching);
      } else {
        // AND: intersect
        const intersected = new Set<number>();
        for (const id of matching) {
          if (result.has(id)) intersected.add(id);
        }
        result = intersected;
      }
    }

    return result ? [...result] : null;
  }

  private matchFilter(filter: FilterOp): number[] | null {
    switch (filter.op) {
      case "eq": {
        // Only safe for exact-partition data (min===max per page).
        // For range data, a value between min and max wouldn't be indexed.
        if (!this.exactPartition) return null;
        const entry = this.index.get(partKey(filter.value));
        return entry ?? [];
      }
      case "in": {
        if (!this.exactPartition) return null;
        if (!Array.isArray(filter.value)) return null;
        const ids = new Set<number>();
        for (const v of filter.value) {
          const entry = this.index.get(partKey(v));
          if (entry) for (const id of entry) ids.add(id);
        }
        return [...ids];
      }
      case "neq": {
        if (!this.exactPartition) return null;
        const excluded = new Set(this.index.get(partKey(filter.value)) ?? []);
        return this.allFragmentIds.filter(id => !excluded.has(id));
      }
      case "not_in": {
        if (!this.exactPartition) return null;
        if (!Array.isArray(filter.value)) return null;
        const excluded = new Set<number>();
        for (const v of filter.value) {
          const entry = this.index.get(partKey(v));
          if (entry) for (const id of entry) excluded.add(id);
        }
        return this.allFragmentIds.filter(id => !excluded.has(id));
      }
      default:
        // gt, gte, lt, lte, between, like — can't use exact partition index
        // Fall through to min/max based pruning
        return null;
    }
  }

  /** Serialize to a plain object for DO durable storage. */
  serialize(): { column: string; index: Record<string, number[]>; allFragmentIds: number[]; exactPartition: boolean } {
    const index: Record<string, number[]> = {};
    for (const [key, ids] of this.index) index[key] = ids;
    return { column: this.column, index, allFragmentIds: this.allFragmentIds, exactPartition: this.exactPartition };
  }

  /** Restore from serialized form. */
  static deserialize(data: { column: string; index?: Record<string, number[]>; allFragmentIds?: number[]; exactPartition?: boolean }): PartitionCatalog {
    const catalog = new PartitionCatalog(data.column);
    if (data.index) {
      for (const [key, ids] of Object.entries(data.index)) {
        catalog.index.set(key, ids);
      }
    }
    catalog.allFragmentIds = data.allFragmentIds ?? [];
    catalog.allFragmentIdSet = new Set(catalog.allFragmentIds);
    catalog.exactPartition = data.exactPartition ?? true; // backwards-compat: old catalogs assumed exact
    return catalog;
  }

  /** Stats for diagnostics. */
  stats(): { column: string; partitionValues: number; fragments: number; indexSizeBytes: number } {
    let indexSizeBytes = 0;
    for (const [key, ids] of this.index) {
      indexSizeBytes += key.length * 2 + ids.length * 4; // rough estimate
    }
    return {
      column: this.column,
      partitionValues: this.index.size,
      fragments: this.allFragmentIds.length,
      indexSizeBytes,
    };
  }
}
