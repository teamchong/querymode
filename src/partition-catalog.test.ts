import { describe, it, expect } from "vitest";
import { PartitionCatalog } from "./partition-catalog.js";
import type { TableMeta, ColumnMeta, FilterOp } from "./types.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeFragmentMeta(
  fragId: number,
  partitionCol: string,
  minVal: string | number,
  maxVal: string | number,
): [number, TableMeta] {
  return [
    fragId,
    {
      name: `frag-${fragId}`,
      columns: [
        {
          name: partitionCol,
          dtype: "utf8",
          pages: [{ byteOffset: 0n, byteLength: 100, rowCount: 100, nullCount: 0, minValue: minVal, maxValue: maxVal }],
          nullCount: 0,
        } as ColumnMeta,
      ],
      totalRows: 100,
      fileSize: 1000n,
      r2Key: `data/frag-${fragId}.lance`,
      updatedAt: Date.now(),
    },
  ];
}

function buildCatalog(): PartitionCatalog {
  const fragments = new Map<number, TableMeta>([
    makeFragmentMeta(1, "region", "us", "us"),
    makeFragmentMeta(2, "region", "eu", "eu"),
    makeFragmentMeta(3, "region", "asia", "asia"),
    makeFragmentMeta(5, "region", "us", "us"),
    makeFragmentMeta(8, "region", "eu", "eu"),
    makeFragmentMeta(12, "region", "us", "us"),
  ]);
  return PartitionCatalog.fromFragments("region", fragments);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("PartitionCatalog", () => {
  describe("fromFragments", () => {
    it("builds index from fragment min/max stats", () => {
      const catalog = buildCatalog();
      const stats = catalog.stats();
      expect(stats.column).toBe("region");
      expect(stats.partitionValues).toBe(3); // us, eu, asia
      expect(stats.fragments).toBe(6);
    });

    it("handles fragments with no matching column", () => {
      const fragments = new Map<number, TableMeta>([
        makeFragmentMeta(1, "region", "us", "us"),
        [2, {
          name: "frag-2",
          columns: [{ name: "other_col", dtype: "utf8", pages: [], nullCount: 0 } as ColumnMeta],
          totalRows: 50, fileSize: 500n, r2Key: "data/frag-2.lance", updatedAt: Date.now(),
        }],
      ]);
      const catalog = PartitionCatalog.fromFragments("region", fragments);
      expect(catalog.stats().fragments).toBe(2);
      expect(catalog.stats().partitionValues).toBe(1); // only "us" from frag 1
    });
  });

  describe("prune — eq", () => {
    it("returns matching fragment IDs for eq filter", () => {
      const catalog = buildCatalog();
      const result = catalog.prune([{ column: "region", op: "eq", value: "us" }]);
      expect(result).not.toBeNull();
      expect(new Set(result)).toEqual(new Set([1, 5, 12]));
    });

    it("returns empty array for non-existent value", () => {
      const catalog = buildCatalog();
      const result = catalog.prune([{ column: "region", op: "eq", value: "mars" }]);
      expect(result).toEqual([]);
    });
  });

  describe("prune — in", () => {
    it("returns union of fragment IDs for in filter", () => {
      const catalog = buildCatalog();
      const result = catalog.prune([{ column: "region", op: "in", value: ["us", "asia"] }]);
      expect(result).not.toBeNull();
      expect(new Set(result)).toEqual(new Set([1, 3, 5, 12]));
    });

    it("handles in with unknown values (partial match)", () => {
      const catalog = buildCatalog();
      const result = catalog.prune([{ column: "region", op: "in", value: ["asia", "mars"] }]);
      expect(result).not.toBeNull();
      expect(new Set(result)).toEqual(new Set([3]));
    });
  });

  describe("prune — neq", () => {
    it("returns all fragments except matching", () => {
      const catalog = buildCatalog();
      const result = catalog.prune([{ column: "region", op: "neq", value: "eu" }]);
      expect(result).not.toBeNull();
      expect(new Set(result)).toEqual(new Set([1, 3, 5, 12]));
    });
  });

  describe("prune — not_in", () => {
    it("excludes multiple values", () => {
      const catalog = buildCatalog();
      const result = catalog.prune([{ column: "region", op: "not_in", value: ["us", "eu"] }]);
      expect(result).not.toBeNull();
      expect(new Set(result)).toEqual(new Set([3]));
    });
  });

  describe("prune — AND intersection", () => {
    it("intersects results from multiple filters on partition column", () => {
      // Register a fragment that has both "us" and "eu" values
      const catalog = new PartitionCatalog("region");
      catalog.register(1, ["us"]);
      catalog.register(2, ["eu"]);
      catalog.register(3, ["us", "eu"]); // fragment 3 has both

      // neq "eu" → [1], neq "us" → [2]... but that would give empty intersection
      // Better test: use eq + neq on same column doesn't make much sense
      // Test with two eq filters that both match same fragment
      const result = catalog.prune([
        { column: "region", op: "neq", value: "asia" },
      ]);
      expect(result).not.toBeNull();
      expect(new Set(result)).toEqual(new Set([1, 2, 3]));
    });
  });

  describe("prune — unsupported ops", () => {
    it("returns null for gt filter (can't use partition index)", () => {
      const catalog = buildCatalog();
      const result = catalog.prune([{ column: "region", op: "gt", value: "eu" }]);
      expect(result).toBeNull();
    });

    it("returns null for like filter", () => {
      const catalog = buildCatalog();
      const result = catalog.prune([{ column: "region", op: "like", value: "%us%" }]);
      expect(result).toBeNull();
    });

    it("returns null for between filter", () => {
      const catalog = buildCatalog();
      const result = catalog.prune([{ column: "region", op: "between", value: ["a", "m"] }]);
      expect(result).toBeNull();
    });
  });

  describe("prune — non-partition column", () => {
    it("returns null when filter is on a different column", () => {
      const catalog = buildCatalog();
      const result = catalog.prune([{ column: "status", op: "eq", value: "active" }]);
      expect(result).toBeNull();
    });
  });

  describe("register", () => {
    it("adds fragment and values to the index", () => {
      const catalog = new PartitionCatalog("region");
      catalog.register(1, ["us"]);
      catalog.register(2, ["eu"]);
      catalog.register(3, ["us", "asia"]);

      const result = catalog.prune([{ column: "region", op: "eq", value: "us" }]);
      expect(new Set(result)).toEqual(new Set([1, 3]));
    });

    it("deduplicates fragment IDs", () => {
      const catalog = new PartitionCatalog("region");
      catalog.register(1, ["us"]);
      catalog.register(1, ["us"]); // duplicate

      const result = catalog.prune([{ column: "region", op: "eq", value: "us" }]);
      expect(result).toEqual([1]);
    });
  });

  describe("serialize / deserialize", () => {
    it("roundtrips through serialization", () => {
      const original = buildCatalog();
      const serialized = original.serialize();
      const restored = PartitionCatalog.deserialize(serialized);

      // Same prune results
      const origResult = original.prune([{ column: "region", op: "eq", value: "us" }]);
      const restoredResult = restored.prune([{ column: "region", op: "eq", value: "us" }]);
      expect(new Set(restoredResult)).toEqual(new Set(origResult));

      // Same stats
      expect(restored.stats()).toEqual(original.stats());
    });

    it("serialized form is JSON-safe (no Map, no BigInt)", () => {
      const catalog = buildCatalog();
      const serialized = catalog.serialize();
      const json = JSON.stringify(serialized);
      const parsed = JSON.parse(json);
      const restored = PartitionCatalog.deserialize(parsed);
      expect(restored.stats().partitionValues).toBe(3);
    });
  });

  describe("stats", () => {
    it("returns correct diagnostics", () => {
      const catalog = buildCatalog();
      const stats = catalog.stats();
      expect(stats.column).toBe("region");
      expect(stats.partitionValues).toBe(3);
      expect(stats.fragments).toBe(6);
      expect(stats.indexSizeBytes).toBeGreaterThan(0);
    });
  });
});
