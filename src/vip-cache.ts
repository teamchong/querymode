/**
 * VIP Cache — LRU with frequency-based eviction protection.
 *
 * Hot entries (accessed >= vipThreshold times) are "VIP" and protected
 * from eviction. Only cold entries are evicted when capacity is reached.
 * If all entries are VIP, the least-recently-used VIP is evicted.
 *
 * Inspired by Zell's expert caching with VIP pinning — hot model shards
 * stay in cache even under pressure from cold one-off accesses.
 */
export class VipCache<K, V> {
  private map = new Map<K, { value: V; accessCount: number; lastAccess: number; expiresAt?: number; refCount: number; pendingEviction: boolean }>();
  private maxEntries: number;
  private vipThreshold: number;

  constructor(maxEntries: number, vipThreshold = 3) {
    this.maxEntries = maxEntries;
    this.vipThreshold = vipThreshold;
  }

  get(key: K): V | undefined {
    const entry = this.map.get(key);
    if (!entry) return undefined;
    if (entry.expiresAt && Date.now() > entry.expiresAt) {
      if (entry.refCount === 0) this.map.delete(key);
      return undefined;
    }
    entry.accessCount++;
    entry.lastAccess = Date.now();
    return entry.value;
  }

  /** Like get(), but increments refCount to prevent eviction. Caller must call release() when done. */
  acquire(key: K): V | undefined {
    const entry = this.map.get(key);
    if (!entry) return undefined;
    if (entry.expiresAt && Date.now() > entry.expiresAt) {
      // Don't delete expired entries that are still referenced
      if (entry.refCount > 0) return undefined;
      this.map.delete(key);
      return undefined;
    }
    entry.accessCount++;
    entry.lastAccess = Date.now();
    entry.refCount++;
    return entry.value;
  }

  /** Release a reference acquired via acquire(). If refCount drops to 0 and entry was marked for eviction, delete it. */
  release(key: K): void {
    const entry = this.map.get(key);
    if (!entry) return;
    if (entry.refCount > 0) entry.refCount--;
    if (entry.refCount === 0 && entry.pendingEviction) {
      this.map.delete(key);
    }
  }

  set(key: K, value: V): void {
    const existing = this.map.get(key);
    if (existing) {
      existing.value = value;
      existing.accessCount++;
      existing.lastAccess = Date.now();
      existing.expiresAt = undefined; // clear TTL — set() implies no expiry
      existing.pendingEviction = false; // revive: set() overrides pending deletion
      return;
    }

    if (this.map.size >= this.maxEntries) {
      this.evict();
    }

    this.map.set(key, { value, accessCount: 1, lastAccess: Date.now(), refCount: 0, pendingEviction: false });
  }

  setWithTTL(key: K, value: V, ttlMs: number): void {
    const existing = this.map.get(key);
    if (existing) {
      existing.value = value;
      existing.accessCount++;
      existing.lastAccess = Date.now();
      existing.expiresAt = Date.now() + ttlMs;
      existing.pendingEviction = false; // revive: setWithTTL() overrides pending deletion
      return;
    }

    if (this.map.size >= this.maxEntries) {
      this.evict();
    }

    this.map.set(key, { value, accessCount: 1, lastAccess: Date.now(), expiresAt: Date.now() + ttlMs, refCount: 0, pendingEviction: false });
  }

  invalidateByPrefix(prefix: string): number {
    let count = 0;
    for (const [key, entry] of this.map) {
      if (String(key).startsWith(prefix)) {
        if (entry.refCount > 0) {
          entry.pendingEviction = true;
        } else {
          this.map.delete(key);
        }
        count++;
      }
    }
    return count;
  }

  has(key: K): boolean {
    const entry = this.map.get(key);
    if (!entry) return false;
    if (entry.expiresAt && Date.now() > entry.expiresAt) {
      if (entry.refCount === 0) this.map.delete(key);
      return false;
    }
    return true;
  }

  /** Return the access count for a key without incrementing it. Returns 0 if not present. */
  accessCount(key: K): number {
    return this.map.get(key)?.accessCount ?? 0;
  }

  delete(key: K): boolean {
    const entry = this.map.get(key);
    if (!entry) return false;
    if (entry.refCount > 0) {
      entry.pendingEviction = true;
      return true;
    }
    return this.map.delete(key);
  }

  get size(): number {
    return this.map.size;
  }

  entries(): IterableIterator<[K, { value: V; accessCount: number; lastAccess: number }]> {
    return this.map.entries();
  }

  /** Diagnostics: return VIP vs cold entry counts and locked (refCount > 0) count */
  stats(): { total: number; vip: number; cold: number; vipThreshold: number; lockedCount: number } {
    let vip = 0;
    let lockedCount = 0;
    for (const entry of this.map.values()) {
      if (entry.accessCount >= this.vipThreshold) vip++;
      if (entry.refCount > 0) lockedCount++;
    }
    return { total: this.map.size, vip, cold: this.map.size - vip, vipThreshold: this.vipThreshold, lockedCount };
  }

  private evict(): void {
    // First try to evict any expired entry (with refCount === 0)
    const now = Date.now();
    for (const [key, entry] of this.map) {
      if (entry.expiresAt && now > entry.expiresAt && entry.refCount === 0) {
        this.map.delete(key);
        return;
      }
    }

    // Try to evict coldest (below VIP threshold) entry with refCount === 0
    let coldestKey: K | undefined;
    let coldestTime = Infinity;

    for (const [key, entry] of this.map) {
      if (entry.refCount === 0 && entry.accessCount < this.vipThreshold && entry.lastAccess < coldestTime) {
        coldestTime = entry.lastAccess;
        coldestKey = key;
      }
    }

    if (coldestKey !== undefined) {
      this.map.delete(coldestKey);
      return;
    }

    // All cold entries are locked — evict the least-recently-used VIP with refCount === 0
    let lruKey: K | undefined;
    let lruTime = Infinity;
    for (const [key, entry] of this.map) {
      if (entry.refCount === 0 && entry.lastAccess < lruTime) {
        lruTime = entry.lastAccess;
        lruKey = key;
      }
    }
    if (lruKey !== undefined) {
      this.map.delete(lruKey);
      return;
    }

    // ALL entries have refCount > 0 — do not evict, let the map grow beyond maxEntries temporarily
  }
}
