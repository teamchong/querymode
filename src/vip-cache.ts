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
  private map = new Map<K, { value: V; accessCount: number; lastAccess: number; expiresAt?: number }>();
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
      this.map.delete(key);
      return undefined;
    }
    entry.accessCount++;
    entry.lastAccess = Date.now();
    return entry.value;
  }

  set(key: K, value: V): void {
    const existing = this.map.get(key);
    if (existing) {
      existing.value = value;
      existing.accessCount++;
      existing.lastAccess = Date.now();
      return;
    }

    if (this.map.size >= this.maxEntries) {
      this.evict();
    }

    this.map.set(key, { value, accessCount: 1, lastAccess: Date.now() });
  }

  setWithTTL(key: K, value: V, ttlMs: number): void {
    const existing = this.map.get(key);
    if (existing) {
      existing.value = value;
      existing.accessCount++;
      existing.lastAccess = Date.now();
      existing.expiresAt = Date.now() + ttlMs;
      return;
    }

    if (this.map.size >= this.maxEntries) {
      this.evict();
    }

    this.map.set(key, { value, accessCount: 1, lastAccess: Date.now(), expiresAt: Date.now() + ttlMs });
  }

  invalidateByPrefix(prefix: string): number {
    let count = 0;
    for (const key of this.map.keys()) {
      if (String(key).startsWith(prefix)) {
        this.map.delete(key);
        count++;
      }
    }
    return count;
  }

  has(key: K): boolean {
    return this.map.has(key);
  }

  /** Return the access count for a key without incrementing it. Returns 0 if not present. */
  accessCount(key: K): number {
    return this.map.get(key)?.accessCount ?? 0;
  }

  delete(key: K): boolean {
    return this.map.delete(key);
  }

  get size(): number {
    return this.map.size;
  }

  entries(): IterableIterator<[K, { value: V; accessCount: number; lastAccess: number }]> {
    return this.map.entries();
  }

  /** Diagnostics: return VIP vs cold entry counts */
  stats(): { total: number; vip: number; cold: number; vipThreshold: number } {
    let vip = 0;
    for (const entry of this.map.values()) {
      if (entry.accessCount >= this.vipThreshold) vip++;
    }
    return { total: this.map.size, vip, cold: this.map.size - vip, vipThreshold: this.vipThreshold };
  }

  private evict(): void {
    // First try to evict any expired entry
    const now = Date.now();
    for (const [key, entry] of this.map) {
      if (entry.expiresAt && now > entry.expiresAt) {
        this.map.delete(key);
        return;
      }
    }

    // Try to evict coldest (below VIP threshold) entry first
    let coldestKey: K | undefined;
    let coldestTime = Infinity;

    for (const [key, entry] of this.map) {
      if (entry.accessCount < this.vipThreshold && entry.lastAccess < coldestTime) {
        coldestTime = entry.lastAccess;
        coldestKey = key;
      }
    }

    if (coldestKey !== undefined) {
      this.map.delete(coldestKey);
      return;
    }

    // All entries are VIP — evict the least-recently-used VIP
    let lruKey: K | undefined;
    let lruTime = Infinity;
    for (const [key, entry] of this.map) {
      if (entry.lastAccess < lruTime) {
        lruTime = entry.lastAccess;
        lruKey = key;
      }
    }
    if (lruKey !== undefined) {
      this.map.delete(lruKey);
    }
  }
}
