import type { Env } from "./types.js";

/** Cached shard bucket array — computed once per DO lifetime. */
let cachedAllBuckets: R2Bucket[] | null = null;

/**
 * Resolve R2 bucket for a given key. Routes by FNV-1a hash of the R2 key
 * prefix (table name) across all available buckets.
 *
 * When DATA_BUCKET_1/2/3 are not configured, returns DATA_BUCKET (zero overhead).
 * When configured, distributes tables across buckets for 2-4x rate limit increase.
 */
export function resolveBucket(env: Env, r2Key: string): R2Bucket {
  if (!cachedAllBuckets) {
    const shards = [env.DATA_BUCKET_1, env.DATA_BUCKET_2, env.DATA_BUCKET_3]
      .filter((b): b is R2Bucket => !!b);
    cachedAllBuckets = shards.length > 0
      ? [env.DATA_BUCKET, ...shards]
      : [env.DATA_BUCKET];
  }

  if (cachedAllBuckets.length === 1) return cachedAllBuckets[0];

  const prefix = r2Key.split("/")[0] ?? r2Key;
  let h = 0x811c9dc5;
  for (let i = 0; i < prefix.length; i++) {
    h ^= prefix.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return cachedAllBuckets[(h >>> 0) % cachedAllBuckets.length];
}
