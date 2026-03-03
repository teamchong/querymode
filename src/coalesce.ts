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
