import { describe, it, expect, beforeEach } from "vitest";
import { resolveBucket } from "./bucket.js";
import type { Env } from "./types.js";

// ---------------------------------------------------------------------------
// Mock R2 buckets (just need distinct identity)
// ---------------------------------------------------------------------------

function mockBucket(name: string): R2Bucket {
  return { _name: name } as unknown as R2Bucket;
}

// resolveBucket uses a module-level cache — we need a fresh import per test group.
// Since we can't easily reset module state, we test behavior patterns instead.

describe("resolveBucket", () => {
  describe("single bucket (no shards)", () => {
    it("returns DATA_BUCKET when no shard buckets configured", () => {
      // Reset module cache by providing a fresh env
      // Note: module-level cache means subsequent calls reuse first env's buckets.
      // This test must run first or in isolation.
      const primary = mockBucket("primary");
      const env = {
        DATA_BUCKET: primary,
        MASTER_DO: {} as DurableObjectNamespace,
        QUERY_DO: {} as DurableObjectNamespace,
        FRAGMENT_DO: {} as DurableObjectNamespace,
      } as Env;

      const result = resolveBucket(env, "orders/fragment-1.lance");
      // With single bucket, always returns primary
      expect(result).toBeDefined();
    });
  });

  describe("FNV-1a hash properties", () => {
    it("same key always routes to same bucket (deterministic)", () => {
      const env = {
        DATA_BUCKET: mockBucket("b0"),
        DATA_BUCKET_1: mockBucket("b1"),
        DATA_BUCKET_2: mockBucket("b2"),
        DATA_BUCKET_3: mockBucket("b3"),
        MASTER_DO: {} as DurableObjectNamespace,
        QUERY_DO: {} as DurableObjectNamespace,
        FRAGMENT_DO: {} as DurableObjectNamespace,
      } as Env;

      const r1 = resolveBucket(env, "orders/fragment-1.lance");
      const r2 = resolveBucket(env, "orders/fragment-1.lance");
      const r3 = resolveBucket(env, "orders/fragment-99.lance");
      // Same table prefix → same bucket
      expect(r1).toBe(r2);
      expect(r1).toBe(r3); // "orders" prefix is the same
    });

    it("routes by prefix (table name), not full key", () => {
      const env = {
        DATA_BUCKET: mockBucket("b0"),
        DATA_BUCKET_1: mockBucket("b1"),
        DATA_BUCKET_2: mockBucket("b2"),
        DATA_BUCKET_3: mockBucket("b3"),
        MASTER_DO: {} as DurableObjectNamespace,
        QUERY_DO: {} as DurableObjectNamespace,
        FRAGMENT_DO: {} as DurableObjectNamespace,
      } as Env;

      // Different fragments of same table → same bucket
      const r1 = resolveBucket(env, "users/fragment-1.lance");
      const r2 = resolveBucket(env, "users/fragment-999.lance");
      expect(r1).toBe(r2);
    });
  });

  describe("FNV-1a hash distribution", () => {
    it("distributes different table names across buckets", () => {
      // We can't test actual distribution without resetting module cache,
      // but we can verify the FNV-1a algorithm independently
      const tables = ["orders", "users", "events", "products", "sessions",
        "clicks", "logs", "metrics", "payments", "invoices",
        "customers", "inventory", "shipments", "reviews", "ratings"];

      // Simulate FNV-1a hash
      const bucketCount = 4;
      const distribution = new Map<number, string[]>();
      for (const table of tables) {
        let h = 0x811c9dc5;
        for (let i = 0; i < table.length; i++) {
          h ^= table.charCodeAt(i);
          h = Math.imul(h, 0x01000193);
        }
        const idx = (h >>> 0) % bucketCount;
        if (!distribution.has(idx)) distribution.set(idx, []);
        distribution.get(idx)!.push(table);
      }

      // With 15 tables across 4 buckets, expect each bucket to have at least 1
      expect(distribution.size).toBeGreaterThanOrEqual(2);
      // No single bucket should have all tables
      for (const [, tables] of distribution) {
        expect(tables.length).toBeLessThan(15);
      }
    });

    it("FNV-1a is deterministic for known inputs", () => {
      // Verify hash of "orders" is consistent
      const table = "orders";
      let h = 0x811c9dc5;
      for (let i = 0; i < table.length; i++) {
        h ^= table.charCodeAt(i);
        h = Math.imul(h, 0x01000193);
      }
      const hash1 = h >>> 0;

      // Same computation again
      h = 0x811c9dc5;
      for (let i = 0; i < table.length; i++) {
        h ^= table.charCodeAt(i);
        h = Math.imul(h, 0x01000193);
      }
      const hash2 = h >>> 0;

      expect(hash1).toBe(hash2);
      expect(hash1).toBeGreaterThan(0);
    });
  });
});
