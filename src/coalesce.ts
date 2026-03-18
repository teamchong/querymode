/** Byte range to fetch from R2 */
export interface Range {
  column: string;
  offset: number;
  length: number;
}

/** Merged range containing the original sub-ranges */
export interface CoalescedRange {
  offset: number;
  length: number;
  ranges: Range[];
}

/**
 * Compute optimal coalesce gap from a **pre-sorted** range array.
 * Dense layouts (small gaps between pages) benefit from aggressive merging.
 * Sparse layouts waste bandwidth with large gaps.
 * Returns a gap between 16KB (sparse) and 256KB (dense).
 */
function computeGap(sorted: Range[]): number {
  if (sorted.length < 2) return 64 * 1024;
  // Compute median gap between adjacent ranges
  const gaps: number[] = [];
  for (let i = 1; i < sorted.length; i++) {
    const gap = sorted[i].offset - (sorted[i - 1].offset + sorted[i - 1].length);
    if (gap > 0) gaps.push(gap);
  }
  if (gaps.length === 0) return 256 * 1024; // all contiguous — merge aggressively
  gaps.sort((a, b) => a - b);
  const medianGap = gaps[gaps.length >> 1];
  // Clamp: at least 16KB (avoid too many reads), at most 256KB (avoid wasted bandwidth)
  return Math.max(16 * 1024, Math.min(256 * 1024, medianGap * 2));
}

/** Merge nearby byte ranges into fewer R2 reads. Sorts by offset, merges if gap <= maxGap.
 *  If maxGap is omitted, auto-computes from page density (sorts once instead of twice). */
export function coalesceRanges(ranges: Range[], maxGap?: number): CoalescedRange[] {
  if (ranges.length === 0) return [];
  const sorted = [...ranges].sort((a, b) => a.offset - b.offset);
  const gap = maxGap ?? computeGap(sorted);
  const result: CoalescedRange[] = [];
  let cur: CoalescedRange = { offset: sorted[0].offset, length: sorted[0].length, ranges: [sorted[0]] };

  for (let i = 1; i < sorted.length; i++) {
    const r = sorted[i];
    const curEnd = cur.offset + cur.length;
    if (r.offset <= curEnd + gap) {
      cur.length = Math.max(curEnd, r.offset + r.length) - cur.offset;
      cur.ranges.push(r);
    } else {
      result.push(cur);
      cur = { offset: r.offset, length: r.length, ranges: [r] };
    }
  }
  result.push(cur);
  return result;
}

/** Run async tasks with bounded concurrency (max `limit` in-flight at once). */
export async function fetchBounded<T>(tasks: (() => Promise<T>)[], limit: number): Promise<T[]> {
  const results: T[] = new Array(tasks.length);
  let i = 0;
  async function next(): Promise<void> {
    const idx = i++;
    if (idx >= tasks.length) return;
    results[idx] = await tasks[idx]();
    return next();
  }
  await Promise.all(Array.from({ length: Math.min(limit, tasks.length) }, () => next()));
  return results;
}

/** Retry an async function with exponential backoff. 3 total attempts max. */
export async function withRetry<T>(fn: () => Promise<T>, maxRetries = 2, baseDelayMs = 100): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (attempt < maxRetries) {
        await new Promise(r => setTimeout(r, baseDelayMs * (1 << attempt)));
      }
    }
  }
  throw lastError;
}

/** Race a promise against a timeout. Rejects with an error if the timeout fires first. */
export function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout>;
  return Promise.race([
    promise,
    new Promise<never>((_, reject) => {
      timer = setTimeout(() => reject(new Error(`Timeout after ${ms}ms`)), ms);
    }),
  ]).finally(() => clearTimeout(timer));
}
