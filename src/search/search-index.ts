/**
 * Full-text search index.
 *
 * Builds an inverted index from document text, persists to binary format,
 * and executes BM25-ranked search queries.
 */

import { tokenizeTerms, type TokenizerConfig } from "./tokenizer.js";
import type { PostingList } from "./posting-list.js";
import { intersect, intersectAll, unionAll, getTermFreq, serializePostings, deserializePostings } from "./posting-list.js";
import { bm25TopK, idf, type BM25Config, type ScoredDoc } from "./bm25.js";
import { fuzzyMatchTerms, maxTyposForTerm } from "./fuzzy.js";

// ---------------------------------------------------------------------------
// Index metadata
// ---------------------------------------------------------------------------

export interface SearchIndexMeta {
  column: string;
  tokenizer: string;
  docCount: number;
  avgDocLength: number;
  vocabularySize: number;
  totalTokens: number;
  builtAt: number;
  version: number;
}

// ---------------------------------------------------------------------------
// Search index stats (per-document length)
// ---------------------------------------------------------------------------

const STATS_MAGIC = 0x53464D51; // "QMFS" LE

export function serializeStats(docLengths: Uint16Array, avgDocLength: number, vocabularySize: number): ArrayBuffer {
  const buf = new ArrayBuffer(16 + docLengths.byteLength);
  const view = new DataView(buf);
  view.setUint32(0, STATS_MAGIC, true);
  view.setUint32(4, docLengths.length, true);
  view.setFloat32(8, avgDocLength, true);
  view.setUint32(12, vocabularySize, true);
  new Uint8Array(buf, 16).set(new Uint8Array(docLengths.buffer, docLengths.byteOffset, docLengths.byteLength));
  return buf;
}

export function deserializeStats(buf: ArrayBuffer): { docLengths: Uint16Array; avgDocLength: number; vocabularySize: number; docCount: number } {
  const view = new DataView(buf);
  const magic = view.getUint32(0, true);
  if (magic !== STATS_MAGIC) throw new Error(`Invalid stats magic: 0x${magic.toString(16)}`);
  const docCount = view.getUint32(4, true);
  const avgDocLength = view.getFloat32(8, true);
  const vocabularySize = view.getUint32(12, true);
  const docLengths = new Uint16Array(buf.slice(16, 16 + docCount * 2));
  return { docLengths, avgDocLength, vocabularySize, docCount };
}

// ---------------------------------------------------------------------------
// In-memory search index (built at ingest, queryable immediately)
// ---------------------------------------------------------------------------

export interface SearchIndexConfig {
  column: string;
  tokenizer?: TokenizerConfig;
  bm25?: BM25Config;
}

export class SearchIndex {
  readonly column: string;
  private postings: Map<string, PostingList>;
  private docLengths: Uint16Array;
  private avgDocLength: number;
  private docCount: number;
  private tokenizerConfig: TokenizerConfig;
  private bm25Config: BM25Config;

  private constructor(
    column: string,
    postings: Map<string, PostingList>,
    docLengths: Uint16Array,
    avgDocLength: number,
    docCount: number,
    tokenizerConfig: TokenizerConfig,
    bm25Config: BM25Config,
  ) {
    this.column = column;
    this.postings = postings;
    this.docLengths = docLengths;
    this.avgDocLength = avgDocLength;
    this.docCount = docCount;
    this.tokenizerConfig = tokenizerConfig;
    this.bm25Config = bm25Config;
  }

  /**
   * Build a search index from document texts.
   * Each element in `texts` corresponds to a document with ID = array index.
   */
  static build(texts: (string | null)[], config: SearchIndexConfig): SearchIndex {
    const tokConfig = config.tokenizer ?? {};
    const docCount = texts.length;

    // Phase 1: Tokenize all documents, collect per-term posting entries
    const termDocMap = new Map<string, { docIds: number[]; tfs: number[] }>();
    const docLengths = new Uint16Array(docCount);
    let totalTokens = 0;

    for (let docId = 0; docId < docCount; docId++) {
      const text = texts[docId];
      if (text === null || text === undefined) {
        docLengths[docId] = 0;
        continue;
      }

      const terms = tokenizeTerms(text, tokConfig);
      docLengths[docId] = Math.min(terms.length, 0xFFFF); // cap at uint16 max
      totalTokens += terms.length;

      // Count term frequencies for this document
      const tfMap = new Map<string, number>();
      for (const term of terms) {
        tfMap.set(term, (tfMap.get(term) ?? 0) + 1);
      }

      // Add to posting lists
      for (const [term, tf] of tfMap) {
        let entry = termDocMap.get(term);
        if (!entry) {
          entry = { docIds: [], tfs: [] };
          termDocMap.set(term, entry);
        }
        entry.docIds.push(docId);
        entry.tfs.push(Math.min(tf, 0xFFFF)); // cap at uint16 max
      }
    }

    // Phase 2: Convert to compact PostingList structures
    const postings = new Map<string, PostingList>();
    for (const [term, entry] of termDocMap) {
      postings.set(term, {
        docIds: new Uint32Array(entry.docIds),
        termFreqs: new Uint16Array(entry.tfs),
        docFrequency: entry.docIds.length,
      });
    }

    const avgDocLength = docCount > 0 ? totalTokens / docCount : 0;

    return new SearchIndex(
      config.column,
      postings,
      docLengths,
      avgDocLength,
      docCount,
      tokConfig,
      config.bm25 ?? {},
    );
  }

  /**
   * Restore a search index from serialized binary data.
   */
  static fromBinary(
    column: string,
    postingsBuf: ArrayBuffer,
    statsBuf: ArrayBuffer,
    config?: { tokenizer?: TokenizerConfig; bm25?: BM25Config },
  ): SearchIndex {
    const postings = deserializePostings(postingsBuf);
    const stats = deserializeStats(statsBuf);
    return new SearchIndex(
      column,
      postings,
      stats.docLengths,
      stats.avgDocLength,
      stats.docCount,
      config?.tokenizer ?? {},
      config?.bm25 ?? {},
    );
  }

  /**
   * Serialize the index for storage (R2 or filesystem).
   */
  serialize(): { postings: ArrayBuffer; stats: ArrayBuffer; meta: SearchIndexMeta } {
    return {
      postings: serializePostings(this.postings),
      stats: serializeStats(this.docLengths, this.avgDocLength, this.postings.size),
      meta: {
        column: this.column,
        tokenizer: "standard",
        docCount: this.docCount,
        avgDocLength: this.avgDocLength,
        vocabularySize: this.postings.size,
        totalTokens: this.docLengths.reduce((s, l) => s + l, 0),
        builtAt: Date.now(),
        version: 1,
      },
    };
  }

  /**
   * Search the index. Returns top-K documents ranked by BM25.
   *
   * @param query - Natural language query text
   * @param topK - Number of results to return
   * @param mode - "and" requires all terms match, "or" requires any term
   * @param typoTolerance - Max typos: 0=exact, 1-2=fuzzy (auto-scaled by term length)
   */
  search(query: string, topK: number, mode: "and" | "or" = "and", typoTolerance = 0): SearchResult {
    const queryTerms = tokenizeTerms(query, this.tokenizerConfig);
    if (queryTerms.length === 0) {
      return { hits: [], totalHits: 0, queryTerms: [], metrics: { termsMatched: 0, postingsRead: 0 } };
    }

    // For each query term, collect one posting list (exact or merged fuzzy).
    // Track which doc IDs match only via fuzzy so we can penalize their scores.
    const matchedPostings: PostingList[] = [];
    const matchedTerms: string[] = [];
    const fuzzyOnlyDocs = new Set<number>(); // docIds that match ONLY via fuzzy, not exact
    let postingsRead = 0;

    for (const term of queryTerms) {
      const exactPosting = this.postings.get(term);

      if (typoTolerance > 0) {
        const maxEdits = maxTyposForTerm(term, typoTolerance);
        const fuzzyMatches = maxEdits > 0
          ? fuzzyMatchTerms(term, this.postings.keys(), maxEdits)
          : [];

        // Collect all posting lists for this term (exact + fuzzy variants)
        const termPostings: PostingList[] = [];
        if (exactPosting) termPostings.push(exactPosting);
        for (const fm of fuzzyMatches) {
          const fp = this.postings.get(fm.term);
          if (fp) termPostings.push(fp);
        }

        if (termPostings.length > 0) {
          // Merge into one posting list via union
          const unionIds = unionAll(termPostings);
          const unionTfs = new Uint16Array(unionIds.length);
          for (let i = 0; i < unionIds.length; i++) {
            let maxTf = 0;
            for (const p of termPostings) {
              const tf = getTermFreq(p, unionIds[i]);
              if (tf > maxTf) maxTf = tf;
            }
            unionTfs[i] = maxTf;

            // Track fuzzy-only docs for score penalty
            if (!exactPosting || getTermFreq(exactPosting, unionIds[i]) === 0) {
              fuzzyOnlyDocs.add(unionIds[i]);
            }
          }
          matchedPostings.push({ docIds: unionIds, termFreqs: unionTfs, docFrequency: unionIds.length });
          matchedTerms.push(term);
          postingsRead += unionIds.length;
        }
      } else {
        if (exactPosting) {
          matchedPostings.push(exactPosting);
          matchedTerms.push(term);
          postingsRead += exactPosting.docIds.length;
        }
      }
    }

    if (matchedPostings.length === 0) {
      return { hits: [], totalHits: 0, queryTerms, metrics: { termsMatched: 0, postingsRead: 0 } };
    }

    const candidateDocIds = mode === "and"
      ? intersectAll(matchedPostings)
      : unionAll(matchedPostings);

    if (candidateDocIds.length === 0) {
      return { hits: [], totalHits: 0, queryTerms, metrics: { termsMatched: matchedTerms.length, postingsRead } };
    }

    // Score candidates via BM25
    let hits = bm25TopK(
      candidateDocIds,
      matchedPostings,
      this.docLengths,
      this.avgDocLength,
      this.docCount,
      topK,
      this.bm25Config,
    );

    // Apply 0.8x penalty for docs that matched only via fuzzy (not exact)
    if (fuzzyOnlyDocs.size > 0) {
      hits = hits.map(h => fuzzyOnlyDocs.has(h.docId) ? { docId: h.docId, score: h.score * 0.8 } : h);
      hits.sort((a, b) => b.score - a.score);
    }

    return {
      hits,
      totalHits: candidateDocIds.length,
      queryTerms: matchedTerms,
      metrics: {
        termsMatched: matchedTerms.length,
        postingsRead,
      },
    };
  }

  /**
   * Search with a columnar filter bitmap applied.
   * Only scores documents that are in BOTH the search results AND the filter set.
   */
  searchWithFilter(
    query: string,
    topK: number,
    filterDocIds: Uint32Array,
    mode: "and" | "or" = "and",
  ): SearchResult {
    const queryTerms = tokenizeTerms(query, this.tokenizerConfig);
    if (queryTerms.length === 0) {
      return { hits: [], totalHits: 0, queryTerms: [], metrics: { termsMatched: 0, postingsRead: 0 } };
    }

    const matchedPostings: PostingList[] = [];
    const matchedTerms: string[] = [];
    let postingsRead = 0;

    for (const term of queryTerms) {
      const posting = this.postings.get(term);
      if (posting) {
        matchedPostings.push(posting);
        matchedTerms.push(term);
        postingsRead += posting.docIds.length;
      }
    }

    if (matchedPostings.length === 0) {
      return { hits: [], totalHits: 0, queryTerms, metrics: { termsMatched: 0, postingsRead: 0 } };
    }

    // Intersect search candidates with filter set
    let searchCandidates = mode === "and"
      ? intersectAll(matchedPostings)
      : unionAll(matchedPostings);

    // Apply columnar filter
    const candidateDocIds = intersect(searchCandidates, filterDocIds);

    if (candidateDocIds.length === 0) {
      return { hits: [], totalHits: 0, queryTerms, metrics: { termsMatched: matchedTerms.length, postingsRead } };
    }

    const hits = bm25TopK(
      candidateDocIds,
      matchedPostings,
      this.docLengths,
      this.avgDocLength,
      this.docCount,
      topK,
      this.bm25Config,
    );

    return {
      hits,
      totalHits: candidateDocIds.length,
      queryTerms: matchedTerms,
      metrics: { termsMatched: matchedTerms.length, postingsRead },
    };
  }

  /** Number of unique terms in the index. */
  get vocabularySize(): number { return this.postings.size; }

  /** Number of indexed documents. */
  get documentCount(): number { return this.docCount; }

  /** Average document length in tokens. */
  get averageDocLength(): number { return this.avgDocLength; }

  /** Look up a single term's posting list. */
  getPosting(term: string): PostingList | undefined { return this.postings.get(term); }
}

// ---------------------------------------------------------------------------
// Search result types
// ---------------------------------------------------------------------------

export interface SearchResult {
  hits: ScoredDoc[];
  totalHits: number;
  queryTerms: string[];
  metrics: {
    termsMatched: number;
    postingsRead: number;
  };
}
