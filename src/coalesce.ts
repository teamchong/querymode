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

/** Merge nearby byte ranges into fewer R2 reads. Sorts by offset, merges if gap <= maxGap. */
export function coalesceRanges(ranges: Range[], maxGap: number): CoalescedRange[] {
  if (ranges.length === 0) return [];
  const sorted = [...ranges].sort((a, b) => a.offset - b.offset);
  const result: CoalescedRange[] = [];
  let cur: CoalescedRange = { offset: sorted[0].offset, length: sorted[0].length, ranges: [sorted[0]] };

  for (let i = 1; i < sorted.length; i++) {
    const r = sorted[i];
    const curEnd = cur.offset + cur.length;
    if (r.offset <= curEnd + maxGap) {
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
