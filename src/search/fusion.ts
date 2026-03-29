/**
 * Rank fusion strategies for hybrid search (BM25 + vector).
 *
 * Reciprocal Rank Fusion (RRF): merges two ranked lists by reciprocal rank sum.
 * Weighted fusion: linear combination of normalized scores.
 */

export interface RankedDoc {
  docId: number;
  score: number;
}

/**
 * Reciprocal Rank Fusion — merge multiple ranked result lists.
 *
 * RRF_score(d) = Σ 1 / (k + rank_i(d))
 * where k = 60 (constant), rank_i(d) = 1-based rank of doc d in list i.
 *
 * Docs appearing in only one list still get a score (from that list alone).
 *
 * @param lists - Array of ranked result lists (each sorted by score descending)
 * @param k - RRF constant (default: 60). Higher k reduces the influence of high-ranking docs.
 * @param topK - Number of results to return
 */
export function reciprocalRankFusion(
  lists: RankedDoc[][],
  topK: number,
  k = 60,
): RankedDoc[] {
  const scores = new Map<number, number>();

  for (const list of lists) {
    for (let rank = 0; rank < list.length; rank++) {
      const docId = list[rank].docId;
      const rrfScore = 1 / (k + rank + 1); // rank is 0-based, RRF uses 1-based
      scores.set(docId, (scores.get(docId) ?? 0) + rrfScore);
    }
  }

  // Sort by RRF score descending, take top-K
  const merged = [...scores.entries()]
    .map(([docId, score]) => ({ docId, score }))
    .sort((a, b) => b.score - a.score);

  return merged.slice(0, topK);
}

/**
 * Weighted fusion — linear combination of normalized scores.
 *
 * Normalizes each list's scores to [0, 1], then combines with weights.
 * score(d) = w1 * norm_score1(d) + w2 * norm_score2(d)
 */
export function weightedFusion(
  lists: RankedDoc[][],
  weights: number[],
  topK: number,
): RankedDoc[] {
  const scores = new Map<number, number>();

  for (let i = 0; i < lists.length; i++) {
    const list = lists[i];
    const weight = weights[i] ?? 1;
    if (list.length === 0) continue;

    // Min-max normalization
    const maxScore = list[0].score; // already sorted descending
    const minScore = list[list.length - 1].score;
    const range = maxScore - minScore || 1; // avoid division by zero

    for (const doc of list) {
      const normalized = (doc.score - minScore) / range;
      scores.set(doc.docId, (scores.get(doc.docId) ?? 0) + normalized * weight);
    }
  }

  const merged = [...scores.entries()]
    .map(([docId, score]) => ({ docId, score }))
    .sort((a, b) => b.score - a.score);

  return merged.slice(0, topK);
}
