/**
 * BM25 scoring engine.
 *
 * Okapi BM25 with configurable k1/b parameters.
 * Top-K maintained via min-heap (no full sort).
 */

import type { PostingList } from "./posting-list.js";
import { getTermFreq } from "./posting-list.js";

export interface BM25Config {
  /** Term frequency saturation (default: 1.2) */
  k1?: number;
  /** Document length normalization (default: 0.75) */
  b?: number;
}

export interface ScoredDoc {
  docId: number;
  score: number;
}

/**
 * Compute IDF for a term.
 * IDF(qi) = ln((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)
 */
export function idf(docFrequency: number, totalDocs: number): number {
  return Math.log((totalDocs - docFrequency + 0.5) / (docFrequency + 0.5) + 1);
}

/**
 * Compute BM25 score for a single document against multiple query terms.
 */
export function bm25ScoreDoc(
  docId: number,
  queryPostings: PostingList[],
  queryIdfs: number[],
  docLength: number,
  avgDocLength: number,
  totalDocs: number,
  k1: number,
  b: number,
): number {
  let score = 0;
  for (let i = 0; i < queryPostings.length; i++) {
    const tf = getTermFreq(queryPostings[i], docId);
    if (tf === 0) continue;
    const idfVal = queryIdfs[i];
    const tfNorm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * docLength / avgDocLength));
    score += idfVal * tfNorm;
  }
  return score;
}

/**
 * Score all candidate documents and return top-K by BM25 score.
 *
 * Uses a min-heap of size K so we never sort the full candidate set.
 * Time: O(candidates × terms + candidates × log(K))
 */
export function bm25TopK(
  candidateDocIds: Uint32Array,
  queryPostings: PostingList[],
  docLengths: Uint16Array,
  avgDocLength: number,
  totalDocs: number,
  topK: number,
  config?: BM25Config,
): ScoredDoc[] {
  const k1 = config?.k1 ?? 1.2;
  const b = config?.b ?? 0.75;

  // Pre-compute IDFs for all query terms
  const queryIdfs = queryPostings.map(p => idf(p.docFrequency, totalDocs));

  // Min-heap of size topK (root = smallest score in the heap)
  const heap: ScoredDoc[] = [];

  for (let i = 0; i < candidateDocIds.length; i++) {
    const docId = candidateDocIds[i];
    const docLen = docLengths[docId] ?? 0;
    const score = bm25ScoreDoc(docId, queryPostings, queryIdfs, docLen, avgDocLength, totalDocs, k1, b);

    if (score <= 0) continue;

    if (heap.length < topK) {
      heap.push({ docId, score });
      if (heap.length === topK) heapify(heap);
    } else if (score > heap[0].score) {
      heap[0] = { docId, score };
      siftDown(heap, 0);
    }
  }

  // Sort heap descending by score for final output
  heap.sort((a, b) => b.score - a.score);
  return heap;
}

// ---------------------------------------------------------------------------
// Min-heap operations (root = minimum score)
// ---------------------------------------------------------------------------

function heapify(heap: ScoredDoc[]): void {
  for (let i = (heap.length >>> 1) - 1; i >= 0; i--) {
    siftDown(heap, i);
  }
}

function siftDown(heap: ScoredDoc[], i: number): void {
  const n = heap.length;
  while (true) {
    let smallest = i;
    const left = 2 * i + 1;
    const right = 2 * i + 2;
    if (left < n && heap[left].score < heap[smallest].score) smallest = left;
    if (right < n && heap[right].score < heap[smallest].score) smallest = right;
    if (smallest === i) break;
    const tmp = heap[i];
    heap[i] = heap[smallest];
    heap[smallest] = tmp;
    i = smallest;
  }
}
