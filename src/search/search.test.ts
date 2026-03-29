import { describe, it, expect } from "vitest";
import { tokenize, tokenizeTerms, isPrimarilyCJK } from "./tokenizer.js";
import { intersect, union, intersectAll, getTermFreq, serializePostings, deserializePostings, type PostingList } from "./posting-list.js";
import { idf, bm25TopK, type ScoredDoc } from "./bm25.js";
import { editDistance, fuzzyMatchTerms, maxTyposForTerm } from "./fuzzy.js";
import { reciprocalRankFusion, weightedFusion } from "./fusion.js";
import { stem } from "./stemmer.js";
import { SearchIndex } from "./search-index.js";

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

describe("tokenizer", () => {
  it("splits on whitespace and punctuation", () => {
    expect(tokenizeTerms("hello world")).toEqual(["hello", "world"]);
    expect(tokenizeTerms("hello, world!")).toEqual(["hello", "world"]);
  });

  it("lowercases", () => {
    expect(tokenizeTerms("Hello WORLD")).toEqual(["hello", "world"]);
  });

  it("folds accented characters", () => {
    const terms = tokenizeTerms("café naïve");
    expect(terms).toContain("cafe");
    expect(terms).toContain("naive");
  });

  it("removes English stopwords by default", () => {
    const terms = tokenizeTerms("the quick brown fox is a lazy dog");
    expect(terms).not.toContain("the");
    expect(terms).not.toContain("is");
    expect(terms).not.toContain("a");
    expect(terms).toContain("quick");
    expect(terms).toContain("brown");
    expect(terms).toContain("fox");
    expect(terms).toContain("lazy");
    expect(terms).toContain("dog");
  });

  it("preserves numbers", () => {
    expect(tokenizeTerms("iPhone 14 Pro")).toEqual(["iphone", "14", "pro"]);
  });

  it("handles empty string", () => {
    expect(tokenizeTerms("")).toEqual([]);
  });

  it("handles punctuation-only", () => {
    expect(tokenizeTerms("!!! ???")).toEqual([]);
  });

  it("disables stopwords when configured", () => {
    const terms = tokenizeTerms("the is a", { stopwords: "none" });
    expect(terms).toEqual(["the", "is", "a"]);
  });

  it("splits CJK into bigrams", () => {
    const terms = tokenizeTerms("東京都");
    expect(terms).toEqual(["東京", "京都"]);
  });

  it("handles single CJK character", () => {
    const terms = tokenizeTerms("雨");
    expect(terms).toEqual(["雨"]);
  });

  it("handles mixed Latin and CJK", () => {
    const terms = tokenizeTerms("hello 東京");
    expect(terms).toContain("hello");
    expect(terms).toContain("東京");
  });

  it("returns offsets with tokenize()", () => {
    const tokens = tokenize("hello world");
    expect(tokens[0]).toEqual({ text: "hello", startOffset: 0, endOffset: 5 });
    expect(tokens[1]).toEqual({ text: "world", startOffset: 6, endOffset: 11 });
  });

  it("detects CJK text", () => {
    expect(isPrimarilyCJK("東京都の天気")).toBe(true);
    expect(isPrimarilyCJK("hello world")).toBe(false);
    expect(isPrimarilyCJK("hello 東京")).toBe(false); // 50/50
  });
});

// ---------------------------------------------------------------------------
// Posting Lists
// ---------------------------------------------------------------------------

describe("posting-list", () => {
  it("intersects sorted arrays", () => {
    const a = new Uint32Array([1, 3, 5, 7, 9]);
    const b = new Uint32Array([2, 3, 5, 8, 9]);
    expect([...intersect(a, b)]).toEqual([3, 5, 9]);
  });

  it("unions sorted arrays", () => {
    const a = new Uint32Array([1, 3, 5]);
    const b = new Uint32Array([2, 3, 6]);
    expect([...union(a, b)]).toEqual([1, 2, 3, 5, 6]);
  });

  it("intersects empty arrays", () => {
    expect([...intersect(new Uint32Array([1, 2]), new Uint32Array([]))]).toEqual([]);
  });

  it("looks up term frequency via binary search", () => {
    const posting: PostingList = {
      docIds: new Uint32Array([10, 20, 30, 40, 50]),
      termFreqs: new Uint16Array([1, 3, 2, 5, 1]),
      docFrequency: 5,
    };
    expect(getTermFreq(posting, 30)).toBe(2);
    expect(getTermFreq(posting, 40)).toBe(5);
    expect(getTermFreq(posting, 25)).toBe(0); // not found
  });

  it("round-trips serialization", () => {
    const index = new Map<string, PostingList>();
    index.set("hello", {
      docIds: new Uint32Array([0, 2, 4]),
      termFreqs: new Uint16Array([3, 1, 2]),
      docFrequency: 3,
    });
    index.set("world", {
      docIds: new Uint32Array([1, 2]),
      termFreqs: new Uint16Array([1, 1]),
      docFrequency: 2,
    });

    const buf = serializePostings(index);
    const restored = deserializePostings(buf);

    expect(restored.size).toBe(2);
    expect([...restored.get("hello")!.docIds]).toEqual([0, 2, 4]);
    expect([...restored.get("hello")!.termFreqs]).toEqual([3, 1, 2]);
    expect(restored.get("world")!.docFrequency).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// BM25
// ---------------------------------------------------------------------------

describe("bm25", () => {
  it("computes IDF correctly", () => {
    // Term in 10 of 100 docs
    const val = idf(10, 100);
    expect(val).toBeGreaterThan(0);
    // Term in all docs has near-zero IDF
    const common = idf(100, 100);
    expect(common).toBeLessThan(0.1);
    // Rare term has higher IDF
    const rare = idf(1, 100);
    expect(rare).toBeGreaterThan(val);
  });

  it("returns top-K by score", () => {
    const posting: PostingList = {
      docIds: new Uint32Array([0, 1, 2, 3, 4]),
      termFreqs: new Uint16Array([5, 1, 3, 1, 10]), // doc 4 has highest TF
      docFrequency: 5,
    };
    const docLengths = new Uint16Array([20, 50, 30, 100, 15]);
    const results = bm25TopK(
      posting.docIds,
      [posting],
      docLengths,
      43, // avgDocLength
      100, // totalDocs
      3,   // topK
    );
    expect(results.length).toBe(3);
    // doc 4 should rank highest (tf=10, short doc)
    expect(results[0].docId).toBe(4);
    // All scores positive
    for (const r of results) expect(r.score).toBeGreaterThan(0);
    // Sorted descending
    expect(results[0].score).toBeGreaterThanOrEqual(results[1].score);
    expect(results[1].score).toBeGreaterThanOrEqual(results[2].score);
  });

  it("handles topK larger than candidates", () => {
    const posting: PostingList = {
      docIds: new Uint32Array([0, 1]),
      termFreqs: new Uint16Array([2, 1]),
      docFrequency: 2,
    };
    const results = bm25TopK(
      posting.docIds, [posting],
      new Uint16Array([10, 10]), 10, 100, 100,
    );
    expect(results.length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// SearchIndex (end-to-end)
// ---------------------------------------------------------------------------

describe("SearchIndex", () => {
  const docs = [
    "iPhone 14 Pro wireless charger fast charging",
    "Samsung Galaxy S23 wireless headphones bluetooth",
    "Apple AirPods Pro noise cancelling wireless earbuds",
    "USB-C fast charging cable for iPhone and Samsung",
    "Wireless mouse and keyboard combo for office",
    "iPhone 15 case with MagSafe wireless charging support",
    null, // null doc handled gracefully
  ];

  it("builds and searches", () => {
    const index = SearchIndex.build(docs, { column: "description" });
    const result = index.search("wireless charger", 5);
    expect(result.hits.length).toBeGreaterThan(0);
    expect(result.totalHits).toBeGreaterThan(0);
    expect(result.queryTerms).toEqual(["wireless", "charger"]);
  });

  it("ranks exact match above partial match", () => {
    const index = SearchIndex.build(docs, { column: "description" });
    const result = index.search("iphone charger", 5);
    // Doc 0 and Doc 3 both mention iPhone + charger variants
    expect(result.hits.length).toBeGreaterThan(0);
    // First result should have highest score
    if (result.hits.length > 1) {
      expect(result.hits[0].score).toBeGreaterThanOrEqual(result.hits[1].score);
    }
  });

  it("returns empty for no matches", () => {
    const index = SearchIndex.build(docs, { column: "description" });
    const result = index.search("xyznonexistent", 5);
    expect(result.hits).toEqual([]);
    expect(result.totalHits).toBe(0);
  });

  it("handles empty query", () => {
    const index = SearchIndex.build(docs, { column: "description" });
    const result = index.search("", 5);
    expect(result.hits).toEqual([]);
  });

  it("OR mode returns union of matches", () => {
    const index = SearchIndex.build(docs, { column: "description" });
    const andResult = index.search("iphone samsung", 10, "and");
    const orResult = index.search("iphone samsung", 10, "or");
    // OR should match more docs than AND
    expect(orResult.totalHits).toBeGreaterThanOrEqual(andResult.totalHits);
  });

  it("reports vocabulary and doc stats", () => {
    const index = SearchIndex.build(docs, { column: "description" });
    expect(index.vocabularySize).toBeGreaterThan(10);
    expect(index.documentCount).toBe(7);
    expect(index.averageDocLength).toBeGreaterThan(0);
  });

  it("round-trips serialization", () => {
    const index = SearchIndex.build(docs, { column: "description" });
    const { postings, stats, meta } = index.serialize();

    const restored = SearchIndex.fromBinary("description", postings, stats);
    const result = restored.search("wireless", 5);
    expect(result.hits.length).toBeGreaterThan(0);
    expect(restored.vocabularySize).toBe(index.vocabularySize);
    expect(restored.documentCount).toBe(index.documentCount);
    expect(meta.column).toBe("description");
    expect(meta.docCount).toBe(7);
  });

  it("multi-term AND finds intersection", () => {
    const index = SearchIndex.build(docs, { column: "description" });
    // "wireless" + "noise" + "cancelling" should match doc 2 (AirPods)
    const result = index.search("wireless noise cancelling", 5);
    expect(result.hits.length).toBeGreaterThan(0);
    // Doc 2 is the only one with all three terms
    expect(result.hits[0].docId).toBe(2);
  });

  it("search with filter narrows results", () => {
    const index = SearchIndex.build(docs, { column: "description" });
    // Only allow docs 0, 1, 2
    const filterDocIds = new Uint32Array([0, 1, 2]);
    const result = index.searchWithFilter("wireless", 5, filterDocIds);
    // Should only return from docs 0, 1, 2 — not 4, 5
    for (const hit of result.hits) {
      expect(hit.docId).toBeLessThanOrEqual(2);
    }
  });

  it("handles large doc set efficiently", () => {
    // 10K docs with random text
    const largeDocs: string[] = [];
    const words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"];
    for (let i = 0; i < 10000; i++) {
      const docWords: string[] = [];
      for (let j = 0; j < 20; j++) {
        docWords.push(words[Math.floor(Math.random() * words.length)]);
      }
      largeDocs.push(docWords.join(" "));
    }

    const start = performance.now();
    const index = SearchIndex.build(largeDocs, { column: "text" });
    const buildMs = performance.now() - start;

    const searchStart = performance.now();
    const result = index.search("alpha beta", 10);
    const searchMs = performance.now() - searchStart;

    expect(result.hits.length).toBe(10);
    expect(buildMs).toBeLessThan(5000); // <5s for 10K docs
    expect(searchMs).toBeLessThan(100);  // <100ms search
  });
});

// ---------------------------------------------------------------------------
// Fuzzy Matching
// ---------------------------------------------------------------------------

describe("editDistance", () => {
  it("returns 0 for identical strings", () => {
    expect(editDistance("hello", "hello", 2)).toBe(0);
  });

  it("computes single substitution", () => {
    expect(editDistance("cat", "bat", 2)).toBe(1);
  });

  it("computes single insertion", () => {
    expect(editDistance("cat", "cats", 2)).toBe(1);
  });

  it("computes single deletion", () => {
    expect(editDistance("cats", "cat", 2)).toBe(1);
  });

  it("returns maxDist+1 when distance exceeds limit", () => {
    expect(editDistance("abc", "xyz", 1)).toBe(2); // distance is 3 but capped at maxDist+1=2
  });

  it("handles empty strings", () => {
    expect(editDistance("", "abc", 5)).toBe(3);
    expect(editDistance("abc", "", 5)).toBe(3);
    expect(editDistance("", "", 0)).toBe(0);
  });

  it("early terminates on length difference", () => {
    expect(editDistance("a", "abcde", 2)).toBe(3); // len diff 4 > maxDist 2
  });
});

describe("maxTyposForTerm", () => {
  it("allows 0 typos for short terms (1-4 chars)", () => {
    expect(maxTyposForTerm("cat")).toBe(0);
    expect(maxTyposForTerm("the")).toBe(0);
    expect(maxTyposForTerm("abcd")).toBe(0);
  });

  it("allows 1 typo for medium terms (5-8 chars)", () => {
    expect(maxTyposForTerm("phone")).toBe(1);
    expect(maxTyposForTerm("wireless")).toBe(1);
  });

  it("allows 2 typos for long terms (9+ chars)", () => {
    expect(maxTyposForTerm("bluetooth")).toBe(2);
    expect(maxTyposForTerm("headphones")).toBe(2);
  });

  it("respects configMax as ceiling", () => {
    expect(maxTyposForTerm("bluetooth", 1)).toBe(1); // would be 2, but config caps at 1
    expect(maxTyposForTerm("bluetooth", 0)).toBe(0);
  });
});

describe("fuzzyMatchTerms", () => {
  const vocab = ["iphone", "iphones", "ipbone", "android", "phone", "iphon"];

  it("finds 1-edit matches", () => {
    const matches = fuzzyMatchTerms("iphone", vocab, 1);
    const terms = matches.map(m => m.term);
    expect(terms).toContain("iphon");    // deletion
    expect(terms).toContain("iphones");  // insertion
  });

  it("penalizes first-character mismatch (costs 2 edits)", () => {
    // "aphone" vs "iphone" — first char differs, edit distance is 1, but penalty makes it 3
    const matches = fuzzyMatchTerms("iphone", ["aphone"], 2);
    expect(matches.length).toBe(0); // 1 edit + 2 penalty = 3 > maxEdits 2
  });

  it("returns empty for maxEdits=0", () => {
    expect(fuzzyMatchTerms("iphone", vocab, 0)).toEqual([]);
  });

  it("skips exact match (caller handles it separately)", () => {
    const matches = fuzzyMatchTerms("iphone", vocab, 2);
    expect(matches.map(m => m.term)).not.toContain("iphone");
  });

  it("sorts by distance ascending", () => {
    const matches = fuzzyMatchTerms("iphone", vocab, 2);
    for (let i = 1; i < matches.length; i++) {
      expect(matches[i].distance).toBeGreaterThanOrEqual(matches[i - 1].distance);
    }
  });
});

// ---------------------------------------------------------------------------
// SearchIndex with Typo Tolerance
// ---------------------------------------------------------------------------

describe("SearchIndex typo tolerance", () => {
  const docs = [
    "iPhone 14 Pro wireless charger fast charging",
    "Samsung Galaxy S23 wireless headphones bluetooth",
    "Apple AirPods Pro noise cancelling wireless earbuds",
    "USB-C fast charging cable for iPhone and Samsung",
    "Wireless mouse and keyboard combo for office",
    "iPhone 15 case with MagSafe wireless charging support",
  ];

  it("finds 'iphone' when searching 'iphon' with typoTolerance=1", () => {
    const index = SearchIndex.build(docs, { column: "title" });
    const result = index.search("iphon", 5, "or", 1);
    expect(result.hits.length).toBeGreaterThan(0);
    // Should find docs with "iphone" (1 edit: missing 'e')
    const matchedDocs = result.hits.map(h => h.docId);
    // Docs 0, 3, 5 contain "iphone"
    expect(matchedDocs.some(id => [0, 3, 5].includes(id))).toBe(true);
  });

  it("does NOT match 'cat' for 'dog' (too many edits)", () => {
    const docs2 = ["cat is here", "dog is there"];
    const index = SearchIndex.build(docs2, { column: "text" });
    const result = index.search("dog", 5, "or", 2);
    // "dog" is 3 chars → maxTyposForTerm = 0, so no fuzzy expansion regardless of typoTolerance
    // Only exact match for "dog"
    expect(result.hits.length).toBe(1);
    expect(result.hits[0].docId).toBe(1);
  });

  it("exact match still works with typoTolerance=0", () => {
    const index = SearchIndex.build(docs, { column: "title" });
    const result = index.search("wireless", 5, "or", 0);
    expect(result.hits.length).toBeGreaterThan(0);
  });

  it("typo tolerance does not produce duplicate results", () => {
    const index = SearchIndex.build(docs, { column: "title" });
    const result = index.search("wireless", 5, "or", 2);
    const docIds = result.hits.map(h => h.docId);
    expect(new Set(docIds).size).toBe(docIds.length);
  });

  it("ranks exact match above fuzzy match", () => {
    const docs3 = ["iphone charger cable", "iphon accessories store"];
    const index = SearchIndex.build(docs3, { column: "title" });
    const result = index.search("iphone", 5, "or", 1);
    // Doc 0 has exact "iphone", doc 1 has "iphon" (1 edit away)
    // Doc 0 should score higher
    if (result.hits.length >= 2) {
      const doc0Score = result.hits.find(h => h.docId === 0)?.score ?? 0;
      const doc1Score = result.hits.find(h => h.docId === 1)?.score ?? 0;
      expect(doc0Score).toBeGreaterThan(doc1Score);
    }
  });
});

// ---------------------------------------------------------------------------
// Rank Fusion
// ---------------------------------------------------------------------------

describe("reciprocalRankFusion", () => {
  it("merges two ranked lists", () => {
    const bm25 = [{ docId: 1, score: 10 }, { docId: 2, score: 8 }, { docId: 3, score: 5 }];
    const vector = [{ docId: 2, score: 0.95 }, { docId: 4, score: 0.9 }, { docId: 1, score: 0.85 }];

    const fused = reciprocalRankFusion([bm25, vector], 5);

    // Doc 2 appears in both lists (rank 2 in BM25, rank 1 in vector) — should score highest
    expect(fused[0].docId).toBe(2);
    // Doc 1 appears in both — second highest
    expect(fused[1].docId).toBe(1);
    // All scores positive
    for (const f of fused) expect(f.score).toBeGreaterThan(0);
  });

  it("handles docs appearing in only one list", () => {
    const listA = [{ docId: 1, score: 10 }];
    const listB = [{ docId: 2, score: 5 }];

    const fused = reciprocalRankFusion([listA, listB], 5);
    expect(fused.length).toBe(2);
    // Both should have equal RRF score (both rank 1 in their respective lists)
    expect(fused[0].score).toBe(fused[1].score);
  });

  it("respects topK", () => {
    const list = [{ docId: 1, score: 10 }, { docId: 2, score: 8 }, { docId: 3, score: 5 }];
    const fused = reciprocalRankFusion([list], 2);
    expect(fused.length).toBe(2);
  });

  it("handles empty lists", () => {
    expect(reciprocalRankFusion([], 5)).toEqual([]);
    expect(reciprocalRankFusion([[]], 5)).toEqual([]);
  });
});

describe("weightedFusion", () => {
  it("merges with weights", () => {
    const bm25 = [{ docId: 1, score: 10 }, { docId: 2, score: 5 }];
    const vector = [{ docId: 2, score: 0.95 }, { docId: 3, score: 0.5 }];

    // Heavy vector weight
    const fused = weightedFusion([bm25, vector], [0.3, 0.7], 5);
    // Doc 2 appears in both — should rank highest
    expect(fused[0].docId).toBe(2);
  });

  it("handles single list", () => {
    const list = [{ docId: 1, score: 10 }, { docId: 2, score: 5 }];
    const fused = weightedFusion([list], [1], 5);
    expect(fused[0].docId).toBe(1);
    expect(fused.length).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// Porter Stemmer
// ---------------------------------------------------------------------------

describe("stem", () => {
  it("stems regular plurals", () => {
    expect(stem("cats")).toBe("cat");
    expect(stem("ponies")).toBe("poni");
    expect(stem("caresses")).toBe("caress");
  });

  it("stems -ing forms", () => {
    expect(stem("running")).toBe("run");
    expect(stem("jumping")).toBe("jump");
  });

  it("stems -ed forms", () => {
    expect(stem("walked")).toBe("walk");
    expect(stem("agreed")).toBe("agre");
  });

  it("stems -tion/-ation forms", () => {
    expect(stem("relational")).toBe("relat");
    expect(stem("conditional")).toBe("condit");
  });

  it("preserves short words", () => {
    expect(stem("a")).toBe("a");
    expect(stem("is")).toBe("is");
  });

  it("handles empty string", () => {
    expect(stem("")).toBe("");
  });
});

describe("tokenizer with stemming", () => {
  it("stems tokens when enabled", () => {
    const terms = tokenizeTerms("running cats are jumping", { stemming: true });
    expect(terms).toContain("run");
    expect(terms).toContain("cat");
    expect(terms).toContain("jump");
  });

  it("does not stem when disabled", () => {
    const terms = tokenizeTerms("running cats", { stemming: false });
    expect(terms).toContain("running");
    expect(terms).toContain("cats");
  });

  it("stemming improves recall in search", () => {
    const docs = ["the runners are running fast", "cats and dogs playing", "jumping over fences"];
    const index = SearchIndex.build(docs, { column: "text", tokenizer: { stemming: true } });

    // "run" should match "runners" and "running" via stemming
    const result = index.search("run", 5, "or");
    expect(result.hits.length).toBeGreaterThan(0);
    expect(result.hits[0].docId).toBe(0); // doc with "runners" and "running"
  });
});
