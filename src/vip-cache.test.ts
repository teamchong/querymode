import { describe, it, expect } from "vitest";
import { VipCache } from "./vip-cache.js";

describe("VipCache", () => {
  it("get/set/has basic operations", () => {
    const cache = new VipCache<string, number>(10);
    cache.set("a", 1);
    expect(cache.get("a")).toBe(1);
    expect(cache.has("a")).toBe(true);
    expect(cache.has("b")).toBe(false);
    expect(cache.get("missing")).toBeUndefined();
  });

  it("evicts cold entries before VIP entries", () => {
    const cache = new VipCache<string, number>(3, 3); // capacity=3, vipThreshold=3

    cache.set("cold", 1);  // access 1, oldest lastAccess (inserted first)

    cache.set("hot", 2);
    cache.get("hot"); // access 2
    cache.get("hot"); // access 3 → VIP

    cache.set("warm", 3);
    cache.get("warm"); // access 2 (not VIP, but newer lastAccess than cold)

    // Cache full (3 entries). Insert 4th — should evict "cold" (below VIP, oldest lastAccess)
    cache.set("new", 4);
    expect(cache.has("hot")).toBe(true);  // VIP, protected
    expect(cache.has("warm")).toBe(true); // cold but newer lastAccess
    expect(cache.has("cold")).toBe(false); // evicted (coldest, oldest lastAccess)
    expect(cache.has("new")).toBe(true);
  });

  it("evicts LRU VIP when all entries are VIP", () => {
    const cache = new VipCache<string, number>(2, 1); // vipThreshold=1 → everything is VIP

    cache.set("a", 1); // access 1 → VIP
    cache.set("b", 2); // access 1 → VIP

    // Both are VIP. Insert 3rd — evicts least-recently-used
    cache.set("c", 3);
    expect(cache.size).toBe(2);
    expect(cache.has("a")).toBe(false); // oldest access time
    expect(cache.has("b")).toBe(true);
    expect(cache.has("c")).toBe(true);
  });

  it("stats returns VIP vs cold counts", () => {
    const cache = new VipCache<string, number>(10, 3);
    cache.set("a", 1);
    cache.get("a");
    cache.get("a"); // 3 accesses → VIP
    cache.set("b", 2); // 1 access → cold
    cache.set("c", 3); // 1 access → cold

    const stats = cache.stats();
    expect(stats.total).toBe(3);
    expect(stats.vip).toBe(1);
    expect(stats.cold).toBe(2);
    expect(stats.vipThreshold).toBe(3);
  });

  it("update existing entry preserves access count", () => {
    const cache = new VipCache<string, number>(10, 2);
    cache.set("a", 1);
    cache.set("a", 2); // update bumps access count
    expect(cache.get("a")).toBe(2);
    expect(cache.stats().vip).toBe(1); // 3 accesses (set + set + get) >= threshold 2
  });

  it("delete removes entry", () => {
    const cache = new VipCache<string, number>(10);
    cache.set("a", 1);
    cache.delete("a");
    expect(cache.has("a")).toBe(false);
    expect(cache.size).toBe(0);
  });

  it("entries returns access metadata", () => {
    const cache = new VipCache<string, number>(10, 3);
    cache.set("a", 1);
    cache.get("a");
    cache.get("a");

    const entries = [...cache.entries()];
    expect(entries.length).toBe(1);
    expect(entries[0][0]).toBe("a");
    expect(entries[0][1].value).toBe(1);
    expect(entries[0][1].accessCount).toBe(3);
    expect(entries[0][1].lastAccess).toBeGreaterThan(0);
  });

  it("acquire/release prevents eviction", () => {
    const cache = new VipCache<string, number>(2, 10);
    cache.set("a", 1);
    cache.set("b", 2);

    const val = cache.acquire("a");
    expect(val).toBe(1);

    // Insert 3rd — "a" has refCount=1, should not be evicted
    cache.set("c", 3);
    expect(cache.has("a")).toBe(true);
    expect(cache.has("b")).toBe(false); // "b" evicted instead
    expect(cache.has("c")).toBe(true);

    cache.release("a");
  });

  it("delete with refCount>0 defers deletion", () => {
    const cache = new VipCache<string, number>(10);
    cache.set("a", 1);
    cache.acquire("a");

    cache.delete("a");
    // Entry is pending eviction but still in map until release
    expect(cache.size).toBe(1);

    cache.release("a");
    // Now actually removed
    expect(cache.size).toBe(0);
    expect(cache.has("a")).toBe(false);
  });

  it("invalidateByPrefix defers deletion for acquired entries", () => {
    const cache = new VipCache<string, number>(10);
    cache.set("tbl/a", 1);
    cache.set("tbl/b", 2);
    cache.set("other", 3);

    cache.acquire("tbl/a");
    const count = cache.invalidateByPrefix("tbl/");
    expect(count).toBe(2);

    // "tbl/b" deleted immediately, "tbl/a" deferred
    expect(cache.has("tbl/b")).toBe(false);
    expect(cache.size).toBe(2); // "tbl/a" (pending) + "other"

    cache.release("tbl/a");
    expect(cache.size).toBe(1); // only "other"
  });

  it("set() clears expiresAt from previous setWithTTL()", () => {
    const cache = new VipCache<string, number>(10);
    cache.setWithTTL("a", 1, 1); // 1ms TTL

    // Overwrite with set() — should clear TTL
    cache.set("a", 2);

    // Wait past the original TTL
    const start = Date.now();
    while (Date.now() - start < 5) { /* spin */ }

    // Entry should still be accessible (TTL cleared)
    expect(cache.get("a")).toBe(2);
  });

  it("get() does not delete expired entry with refCount>0", () => {
    const cache = new VipCache<string, number>(10);
    cache.setWithTTL("a", 1, 1); // 1ms TTL
    cache.acquire("a");

    const start = Date.now();
    while (Date.now() - start < 5) { /* spin */ }

    // get() returns undefined (expired) but does not delete (refCount > 0)
    expect(cache.get("a")).toBeUndefined();
    expect(cache.size).toBe(1); // still in map

    cache.release("a");
  });
});
