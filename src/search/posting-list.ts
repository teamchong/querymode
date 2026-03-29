/**
 * Posting list storage for the inverted index.
 *
 * Each term maps to a sorted array of document IDs + per-doc term frequencies.
 * Uses compact sorted arrays (not full roaring bitmaps) for Phase 1 — WASM roaring comes later.
 *
 * Set operations (AND/OR) on sorted arrays are O(n+m) via merge-intersect.
 */

/** A single posting list entry: sorted doc IDs + term frequencies. */
export interface PostingList {
  /** Sorted document IDs (ascending) */
  docIds: Uint32Array;
  /** Term frequency for each doc (parallel to docIds) */
  termFreqs: Uint16Array;
  /** Total documents containing this term */
  docFrequency: number;
}

/** Intersect two sorted Uint32Arrays — returns docs present in BOTH. */
export function intersect(a: Uint32Array, b: Uint32Array): Uint32Array {
  const result: number[] = [];
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      result.push(a[i]);
      i++; j++;
    } else if (a[i] < b[j]) {
      i++;
    } else {
      j++;
    }
  }
  return new Uint32Array(result);
}

/** Union two sorted Uint32Arrays — returns docs present in EITHER. */
export function union(a: Uint32Array, b: Uint32Array): Uint32Array {
  const result: number[] = [];
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      result.push(a[i]);
      i++; j++;
    } else if (a[i] < b[j]) {
      result.push(a[i]);
      i++;
    } else {
      result.push(b[j]);
      j++;
    }
  }
  while (i < a.length) result.push(a[i++]);
  while (j < b.length) result.push(b[j++]);
  return new Uint32Array(result);
}

/** Intersect multiple posting lists. Returns doc IDs present in ALL lists. */
export function intersectAll(lists: PostingList[]): Uint32Array {
  if (lists.length === 0) return new Uint32Array(0);
  if (lists.length === 1) return lists[0].docIds;
  // Sort by size ascending — smallest first for early termination
  const sorted = [...lists].sort((a, b) => a.docIds.length - b.docIds.length);
  let result = sorted[0].docIds;
  for (let i = 1; i < sorted.length; i++) {
    result = intersect(result, sorted[i].docIds);
    if (result.length === 0) break;
  }
  return result;
}

/** Union multiple posting lists. Returns doc IDs present in ANY list. */
export function unionAll(lists: PostingList[]): Uint32Array {
  if (lists.length === 0) return new Uint32Array(0);
  if (lists.length === 1) return lists[0].docIds;
  let result = lists[0].docIds;
  for (let i = 1; i < lists.length; i++) {
    result = union(result, lists[i].docIds);
  }
  return result;
}

/**
 * Look up term frequency for a specific docId in a posting list.
 * Uses binary search since docIds are sorted.
 */
export function getTermFreq(posting: PostingList, docId: number): number {
  const ids = posting.docIds;
  let lo = 0, hi = ids.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (ids[mid] === docId) return posting.termFreqs[mid];
    if (ids[mid] < docId) lo = mid + 1;
    else hi = mid - 1;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Serialization — write/read posting lists to/from binary for R2 storage
// ---------------------------------------------------------------------------

const POSTINGS_MAGIC = 0x50514D51; // "QMQP" LE

/**
 * Serialize a map of term → PostingList into a single binary buffer.
 *
 * Layout:
 *   [4] magic
 *   [4] term count
 *   Per term:
 *     [4] term byte length
 *     [N] term UTF-8
 *     [4] doc frequency
 *     [4] docIds byte length (df * 4)
 *     [M] docIds (uint32 LE)
 *     [4] termFreqs byte length (df * 2)
 *     [K] termFreqs (uint16 LE)
 */
export function serializePostings(index: Map<string, PostingList>): ArrayBuffer {
  const encoder = new TextEncoder();
  // Pre-calculate total size
  let totalSize = 8; // magic + term count
  const termEntries: { termBytes: Uint8Array; posting: PostingList }[] = [];

  for (const [term, posting] of index) {
    const termBytes = encoder.encode(term);
    termEntries.push({ termBytes, posting });
    totalSize += 4 + termBytes.length + 4 + 4 + posting.docIds.byteLength + 4 + posting.termFreqs.byteLength;
  }

  const buf = new ArrayBuffer(totalSize);
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);
  let offset = 0;

  view.setUint32(offset, POSTINGS_MAGIC, true); offset += 4;
  view.setUint32(offset, termEntries.length, true); offset += 4;

  for (const { termBytes, posting } of termEntries) {
    view.setUint32(offset, termBytes.length, true); offset += 4;
    u8.set(termBytes, offset); offset += termBytes.length;
    view.setUint32(offset, posting.docFrequency, true); offset += 4;

    const docIdBytes = posting.docIds.byteLength;
    view.setUint32(offset, docIdBytes, true); offset += 4;
    u8.set(new Uint8Array(posting.docIds.buffer, posting.docIds.byteOffset, docIdBytes), offset);
    offset += docIdBytes;

    const tfBytes = posting.termFreqs.byteLength;
    view.setUint32(offset, tfBytes, true); offset += 4;
    u8.set(new Uint8Array(posting.termFreqs.buffer, posting.termFreqs.byteOffset, tfBytes), offset);
    offset += tfBytes;
  }

  return buf;
}

/**
 * Deserialize a binary buffer into a term → PostingList map.
 */
export function deserializePostings(buf: ArrayBuffer): Map<string, PostingList> {
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);
  const decoder = new TextDecoder();
  let offset = 0;

  const magic = view.getUint32(offset, true); offset += 4;
  if (magic !== POSTINGS_MAGIC) {
    throw new Error(`Invalid postings magic: 0x${magic.toString(16)}`);
  }

  const termCount = view.getUint32(offset, true); offset += 4;
  const index = new Map<string, PostingList>();

  for (let i = 0; i < termCount; i++) {
    const termLen = view.getUint32(offset, true); offset += 4;
    const term = decoder.decode(u8.subarray(offset, offset + termLen));
    offset += termLen;

    const docFrequency = view.getUint32(offset, true); offset += 4;

    const docIdBytes = view.getUint32(offset, true); offset += 4;
    const docIds = new Uint32Array(buf.slice(offset, offset + docIdBytes));
    offset += docIdBytes;

    const tfBytes = view.getUint32(offset, true); offset += 4;
    const termFreqs = new Uint16Array(buf.slice(offset, offset + tfBytes));
    offset += tfBytes;

    index.set(term, { docIds, termFreqs, docFrequency });
  }

  return index;
}
