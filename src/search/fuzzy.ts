/**
 * Fuzzy matching for typo-tolerant search.
 *
 * Levenshtein edit distance with early termination + vocabulary scan.
 * Phase 1: brute-force scan over vocabulary Map keys.
 * Phase 2 (future): Levenshtein automaton walk over FST.
 */

/**
 * Compute Levenshtein edit distance between two strings.
 * Uses a single-row DP with early termination when distance exceeds maxDist.
 * Returns maxDist+1 if the actual distance exceeds the limit.
 */
export function editDistance(a: string, b: string, maxDist: number): number {
  const m = a.length;
  const n = b.length;

  // Length difference already exceeds max — no need to compute
  if (Math.abs(m - n) > maxDist) return maxDist + 1;

  // Single-row DP: prev[j] = distance between a[0..i-1] and b[0..j-1]
  const prev = new Uint8Array(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;

  for (let i = 1; i <= m; i++) {
    let diagPrev = prev[0];
    prev[0] = i;
    let rowMin = prev[0];

    for (let j = 1; j <= n; j++) {
      const diag = diagPrev;
      diagPrev = prev[j];

      let cost: number;
      if (a.charCodeAt(i - 1) === b.charCodeAt(j - 1)) {
        cost = diag;
      } else {
        cost = Math.min(diag, prev[j], prev[j - 1]) + 1;
      }
      prev[j] = cost;
      if (cost < rowMin) rowMin = cost;
    }

    // Early termination: if the minimum in this row exceeds maxDist, we can't recover
    if (rowMin > maxDist) return maxDist + 1;
  }

  return prev[n];
}

/**
 * Determine max allowed typos for a query term based on its length.
 * - 1-4 chars: 0 typos (too short, too many false matches)
 * - 5-8 chars: 1 typo
 * - 9+ chars: 2 typos
 */
export function maxTyposForTerm(term: string, configMax?: number): number {
  const len = term.length;
  let allowed: number;
  if (len <= 4) allowed = 0;
  else if (len <= 8) allowed = 1;
  else allowed = 2;
  // User config can reduce but not increase beyond the length-based limit
  if (configMax !== undefined && configMax < allowed) allowed = configMax;
  return allowed;
}

export interface FuzzyMatch {
  /** The matching term from the vocabulary */
  term: string;
  /** Edit distance from the query term */
  distance: number;
}

/**
 * Find all terms in a vocabulary within edit distance of the query term.
 * First-character mismatch counts as 2 edits (drastically prunes the match space).
 *
 * @param queryTerm - The (normalized) query term to fuzzy-match
 * @param vocabulary - Iterable of known terms (e.g., Map.keys())
 * @param maxEdits - Maximum allowed edit distance (0, 1, or 2)
 * @returns All matching terms sorted by distance (exact matches first)
 */
export function fuzzyMatchTerms(
  queryTerm: string,
  vocabulary: Iterable<string>,
  maxEdits: number,
): FuzzyMatch[] {
  if (maxEdits === 0) return []; // Exact only — caller handles exact lookup separately

  const matches: FuzzyMatch[] = [];
  const firstChar = queryTerm.charCodeAt(0);

  for (const term of vocabulary) {
    if (term === queryTerm) continue; // Skip exact match — caller already handles it

    // First-character penalty: if first char differs, costs 2 edits.
    // Skip if that already exceeds our budget.
    const firstCharMatch = term.charCodeAt(0) === firstChar;
    const effectiveMax = firstCharMatch ? maxEdits : maxEdits - 2;
    if (effectiveMax < 0) continue;

    const dist = editDistance(queryTerm, term, effectiveMax);
    if (dist <= effectiveMax) {
      // Adjust reported distance: first-char mismatch adds 2 to the penalty
      const reportedDist = firstCharMatch ? dist : dist + 2;
      if (reportedDist <= maxEdits) {
        matches.push({ term, distance: reportedDist });
      }
    }
  }

  // Sort: lower distance first, then alphabetical for stability
  matches.sort((a, b) => a.distance - b.distance || a.term.localeCompare(b.term));
  return matches;
}
