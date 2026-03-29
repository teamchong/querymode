# Spec: Full-Text Search

Status: **Draft** — ready for review before implementation.

## Problem

QueryMode handles OLAP and vector search. It cannot do text search — the thing every exec immediately understands. Typing three letters and getting instant ranked results is what sells a query engine. OLAP demos don't move executives. Search demos do.

Meilisearch, Typesense, and Algolia own this space. They're fast, polished, and developer-friendly. But they're all **single-region, always-on servers** that can't do analytics, can't run at the edge, and charge per-search or require dedicated infra. QueryMode can beat them by being the first engine that does **OLAP + full-text search + vector search in one serverless edge deployment**.

## Non-Goals

- Replace Elasticsearch for log analytics (different workload shape)
- Build a crawler or web scraper
- Build InstantSearch UI libraries (adapter later, not now)
- Support 30+ language tokenizers on day one (English + CJK first)
- Real-time indexing under 10ms (sub-second is fine, matches Turbopuffer)

---

## 1. User-Facing API

### DataFrame API

```typescript
// Basic text search — returns rows ranked by BM25
const results = await qm
  .table("products")
  .search("iphone charger", { typoTolerance: 1 })
  .limit(10)
  .collect()

// Search + filter (the power combo)
const results = await qm
  .table("products")
  .search("wireless headphones")
  .filter("category", "eq", "Electronics")
  .filter("price", "lt", 100)
  .limit(20)
  .collect()

// Hybrid search — BM25 + vector in one query
const results = await qm
  .table("products")
  .search("comfortable noise cancelling")
  .nearestTo("embedding", queryVec, 50, { metric: "cosine" })
  .limit(10)
  .collect()
// Rank fusion: reciprocal rank fusion (RRF) by default

// Faceted search
const results = await qm
  .table("products")
  .search("laptop")
  .facets(["brand", "category", "price_range"])
  .limit(20)
  .exec()
// result.facets = { brand: { "Apple": 12, "Dell": 8, ... }, ... }

// Search with field weighting
const results = await qm
  .table("products")
  .search("iphone", { fields: { title: 3, description: 1 } })
  .limit(10)
  .collect()
```

### SQL Interface

```sql
-- Basic search
SELECT * FROM products WHERE MATCH('iphone charger') LIMIT 10

-- Search with filters
SELECT * FROM products
WHERE MATCH('wireless headphones') AND category = 'Electronics' AND price < 100
LIMIT 20

-- Hybrid search
SELECT * FROM products
WHERE MATCH('comfortable noise cancelling')
  AND NEAR(embedding, [0.1, 0.2, ...], 50, 'cosine')
LIMIT 10

-- Search with field weights
SELECT * FROM products WHERE MATCH('iphone', title^3, description^1) LIMIT 10

-- Faceted search
SELECT * FROM products WHERE MATCH('laptop')
FACET brand, category, price_range
LIMIT 20
```

### HTTP API

```
POST /search
{
  "table": "products",
  "query": "iphone charger",
  "filters": [{ "column": "category", "op": "eq", "value": "Electronics" }],
  "facets": ["brand", "category"],
  "limit": 10,
  "typoTolerance": 1,
  "fields": { "title": 3, "description": 1 }
}

Response:
{
  "rows": [...],
  "rowCount": 10,
  "totalHits": 847,
  "facets": {
    "brand": { "Apple": 42, "Samsung": 38, ... },
    "category": { "Cables": 123, "Cases": 89, ... }
  },
  "durationMs": 12,
  "searchMetrics": {
    "termsMatched": 2,
    "typosApplied": 0,
    "indexSegmentsSearched": 3,
    "postingsRead": 12847
  }
}
```

### Special Result Columns

| Column | Type | Description |
|--------|------|-------------|
| `_score` | float64 | BM25 relevance score (higher = more relevant) |
| `_matched_terms` | string | Comma-separated terms that matched |
| `_highlights` | string | JSON object with highlighted snippets (optional, opt-in) |

---

## 2. Inverted Index

### Data Structure

The inverted index is a new index type stored alongside Lance fragments on R2.

```
{dataset}.lance/
├── _versions/
├── data/
│   ├── {uuid1}.lance            # Row data (existing)
│   └── {uuid2}.lance
├── _index/
│   ├── {column}.fst             # Term dictionary (FST)
│   ├── {column}.postings        # Posting lists (roaring bitmaps)
│   ├── {column}.stats           # Per-document stats (doc length, term count)
│   └── {column}.meta            # Index metadata (vocabulary size, avg doc length, etc.)
```

### Term Dictionary (FST)

Finite State Transducer mapping terms to posting list offsets.

```
Term (UTF-8) → { postingListOffset: uint64, docFrequency: uint32 }
```

- Stored as a compiled FST (same algorithm as BurntSushi's `fst` crate)
- Supports: exact lookup, prefix enumeration, Levenshtein automaton walk
- Compressed size: typically 3-5x smaller than raw sorted term list
- Built in Zig WASM at index time

### Posting List Format

```
Header:
  [4 bytes] Term count (uint32 LE)

Per term:
  [4 bytes] Term length (uint32 LE)
  [N bytes] Term string (UTF-8)
  [4 bytes] Doc frequency (uint32 LE)
  [4 bytes] Posting data length (uint32 LE)
  [M bytes] Roaring bitmap (serialized doc IDs)
  [4 bytes] Term frequency data length (uint32 LE)
  [K bytes] Term frequencies (packed uint16 per doc, indexed by roaring bitmap iteration order)
  [4 bytes] Positions data length (uint32 LE) — 0 if positions disabled
  [P bytes] Positions (per-doc: [uint16 count][uint16 positions...])
```

### Stats File

```
Header:
  [4 bytes] Magic: "QMFS" (QueryMode Full-text Stats)
  [4 bytes] Doc count (uint32 LE)
  [4 bytes] Avg doc length as float32
  [4 bytes] Total terms (vocabulary size, uint32 LE)

Per document (docCount entries):
  [2 bytes] Document length in tokens (uint16 LE)
```

### Index Metadata

```json
{
  "column": "description",
  "tokenizer": "standard",
  "docCount": 1000000,
  "avgDocLength": 42.7,
  "vocabularySize": 87432,
  "totalTokens": 42700000,
  "builtAt": 1711500000,
  "version": 1,
  "config": {
    "typoTolerance": true,
    "minWordLength": { "oneTypo": 5, "twoTypos": 9 },
    "storePositions": false,
    "stopwords": "english"
  }
}
```

---

## 3. Tokenization

### Standard Tokenizer (default)

```
Input:  "The iPhone 14 Pro's battery-life is amazing!!! 🔋"
Output: ["the", "iphone", "14", "pro", "s", "battery", "life", "is", "amazing"]
```

Steps:
1. **Unicode segmentation**: Split on whitespace + punctuation (keep numbers)
2. **Lowercase**: ASCII + Unicode lowercase
3. **ASCII folding**: cafe with accent → cafe, naive with diaeresis → naive
4. **Stopword removal** (optional, configurable): "the", "is", "a", "an", "and", "or", ...

### CJK Tokenizer

```
Input:  "東京都の天気は晴れです"
Output: ["東京", "京都", "都の", "の天", "天気", "気は", "は晴", "晴れ", "れで", "です"]
```

Bigram sliding window. No dictionary required. Handles Chinese, Japanese, Korean.

### Implementation

- Built in **Zig WASM** — runs at index time and query time
- Exposed as WASM exports: `tokenize(textPtr, textLen, outputPtr) → tokenCount`
- Token output: packed `[offset: uint32, length: uint16]` pairs into output buffer
- Tokenizer selection: auto-detect by Unicode script range, or configured per-column

---

## 4. BM25 Scoring

### Formula

```
score(q, d) = Σ over query terms qi:
  IDF(qi) × (tf(qi, d) × (k1 + 1)) / (tf(qi, d) + k1 × (1 - b + b × |d| / avgdl))

where:
  IDF(qi) = ln((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)
  tf(qi, d) = term frequency of qi in document d
  |d| = document length in tokens
  avgdl = average document length across corpus
  N = total document count
  k1 = 1.2 (default, configurable)
  b = 0.75 (default, configurable)
```

### Implementation

- Computed in **Zig WASM** using SIMD for batch scoring
- Input: posting list doc IDs + term frequencies + doc lengths
- Output: scored (docId, score) pairs, top-K maintained via min-heap
- WASM export: `bm25Score(postingsPtr, tfPtr, docLenPtr, count, idf, k1, b, avgdl) → scoresPtr`

### Multi-term Queries

1. Tokenize query into terms
2. For each term: look up posting list in FST, retrieve roaring bitmap of doc IDs
3. Intersect bitmaps (AND for required terms) or union (OR for optional)
4. For each candidate doc: sum BM25 scores across matched terms
5. Top-K via min-heap (no full sort)

---

## 5. Typo Tolerance

### Algorithm

Walk a Levenshtein automaton over the FST. Single pass — O(|FST| × states), not O(|terms| × edit distance).

### Rules

| Query term length | Max typos allowed |
|-------------------|-------------------|
| 1-4 characters | 0 (exact match only) |
| 5-8 characters | 1 |
| 9+ characters | 2 |

First-character error counts as 2 typos (drastically prunes search space).

### Implementation

- **Levenshtein automaton**: Built in Zig. States represent (position, errors) pairs.
- **FST walk**: Automaton and FST are traversed simultaneously. Each FST transition is tested against automaton transitions.
- Produces all matching terms within edit distance, with their posting list offsets.
- WASM export: `fstFuzzySearch(fstPtr, fstLen, queryPtr, queryLen, maxEdits, outputPtr) → matchCount`

### Prefix Search

- Only the **last** query term uses prefix matching (all prior terms must match fully)
- Walk FST from prefix position, enumerate all reachable terminal states
- Prefix results scored with a 0.8x penalty to prefer exact matches

---

## 6. Faceted Search

### Behavior

Facets are computed alongside search results in a **single pass** (not a second query).

```typescript
// Request
.search("laptop").facets(["brand", "category"])

// Response includes:
{
  facets: {
    brand: { "Apple": 42, "Dell": 38, "Lenovo": 31, ... },
    category: { "Laptops": 89, "Accessories": 12, ... }
  }
}
```

### Implementation

1. BM25 search produces candidate doc IDs (roaring bitmap)
2. For each facet column: scan column values for candidate docs, count per value
3. Both steps run in WASM — facet counting is a GROUP BY COUNT on the bitmap

### Disjunctive Faceting

When a user selects a facet value (e.g., brand = "Apple"), facet counts for OTHER values in the same facet group should NOT be filtered out. Only cross-facet filters apply.

---

## 7. Hybrid Search (BM25 + Vector)

### Rank Fusion

When both `.search()` and `.nearestTo()` are present in a query:

1. Execute BM25 search, produce ranked list with scores
2. Execute vector ANN search, produce ranked list with distances
3. Fuse via **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
where k = 60 (constant), rank_i(d) = rank of doc d in result list i
```

4. Sort by RRF score descending, apply limit

### Configurable Fusion

```typescript
.search("laptop")
.nearestTo("embedding", vec, 50)
.fusionStrategy("rrf", { k: 60 })    // default
// or
.fusionStrategy("weighted", { bm25Weight: 0.3, vectorWeight: 0.7 })
```

---

## 8. Index Building

### When

Inverted index is built at **ingest time** (same as vector index).

```typescript
// Create table with search index
await qm.table("products")
  .createIndex("description", { type: "fulltext", tokenizer: "standard" })

// Or during append with auto-indexing
await qm.table("products").append(rows, {
  searchIndexes: ["title", "description"]
})
```

### Build Process

1. `MasterDO.appendRpc()` receives rows
2. Build Lance fragment (existing path)
3. For each indexed column:
   a. Tokenize all values via WASM — produce term stream
   b. Build posting lists (term to doc IDs + term frequencies)
   c. Build FST from sorted term dictionary
   d. Compute per-document stats (length, term count)
   e. Write `_index/{column}.fst`, `.postings`, `.stats`, `.meta` to R2
4. Update manifest with index metadata

### Incremental Index

When new fragments are appended, each fragment gets its own index segment. At query time, results from all segments are merged. Periodic compaction merges segments (background, non-blocking).

---

## 9. Query Execution Path

```
User: .search("iphone charger").filter("price", "lt", 50).limit(10)
  │
  ▼
1. Tokenize query: ["iphone", "charger"]
  │
  ▼
2. Load index:
   a. FST from R2 (_index/{column}.fst) — cached in WASM buffer pool
   b. Stats from R2 (_index/{column}.stats) — cached
  │
  ▼
3. Term lookup:
   a. "iphone": FST lookup → posting list offset → load postings → roaring bitmap
   b. "charger": same
   c. If typo tolerance: walk Levenshtein automaton on FST for each term
  │
  ▼
4. Set operations:
   a. AND: intersect bitmaps → candidate doc IDs
   b. Apply columnar filters: price < 50 → intersect with candidate bitmap
      (Uses existing page-level pruning on the price column)
  │
  ▼
5. BM25 scoring:
   a. Load term frequencies for candidate docs
   b. WASM SIMD batch scoring → (docId, score) pairs
   c. Top-K via min-heap
  │
  ▼
6. Materialize:
   a. Fetch row data for top-K doc IDs from Lance fragments
   b. Attach _score, _matched_terms columns
   c. Return QueryResult
```

### Fan-out for Multi-Fragment Datasets

Same pattern as OLAP fan-out:
- Each Fragment DO searches its own index segment
- Returns top-K from its segment (QMCB with _score column)
- QueryDO merges: k-way merge on _score, re-apply global limit

---

## 10. Acceptance Criteria

### Correctness

- [ ] BM25 scores match reference implementation (plus or minus 0.001) on standardized test corpus
- [ ] Typo tolerance finds "iphon" when searching for "iphone" (edit distance 1)
- [ ] Typo tolerance does NOT match "cat" for "dog" (edit distance 3, exceeds max)
- [ ] Empty query returns all documents (browse mode, no text filter applied)
- [ ] Filters combine with search: `MATCH('x') AND price < 50` returns only matching, cheap results
- [ ] Facet counts are accurate for the filtered+searched result set
- [ ] Hybrid search (BM25 + vector) returns results from both signals, fused correctly
- [ ] CJK text tokenized via bigram, searchable in the same index

### Performance

- [ ] **10M documents, warm cache**: p50 < 50ms, p99 < 200ms (single search, 10 results)
- [ ] **10M documents, cold cache**: p50 < 200ms, p99 < 500ms
- [ ] **Indexing throughput**: > 100K documents/second on 4 vCPU
- [ ] **Index size overhead**: < 2x raw text column size (posting lists + FST + stats)
- [ ] **Memory**: Index search uses < 10MB WASM heap per query (streaming, not loading all postings)

### Integration

- [ ] `.search()` works in DataFrame API (local + edge)
- [ ] `WHERE MATCH(...)` works in SQL
- [ ] Search results stream correctly via PG wire
- [ ] Hybrid search combines `.search()` + `.nearestTo()` in single query
- [ ] Explain output shows: terms matched, postings read, segments searched

---

## 11. Implementation Phases

### Phase 1: Inverted Index + BM25 (foundation)

Build the core search infrastructure.

**Zig WASM (wasm/src/search/):**
- `tokenizer.zig` — Standard tokenizer (Unicode split, lowercase, ASCII fold)
- `fst.zig` — FST construction and lookup
- `posting_list.zig` — Roaring bitmap posting lists with term frequencies
- `bm25.zig` — BM25 scoring with SIMD batch computation
- `search_engine.zig` — Orchestrator: tokenize, FST lookup, intersect, score, top-K

**TypeScript:**
- `src/search-index.ts` — Index building at ingest time (calls WASM)
- `src/search-operator.ts` — SearchOperator for the pull pipeline
- Update `master-do.ts` — Build index during append
- Update `client.ts` — `.search()` method on DataFrame
- Update `sql/compiler.ts` — `MATCH()` syntax

**Tests:**
- BM25 scoring accuracy vs reference
- Tokenization edge cases (Unicode, CJK, empty strings, very long strings)
- End-to-end: ingest then index then search then verify results

### Phase 2: Typo Tolerance

- `wasm/src/search/levenshtein.zig` — Levenshtein automaton construction
- `wasm/src/search/fst_fuzzy.zig` — FST and automaton simultaneous traversal
- Prefix search on last query term
- Config: enable/disable per column, min word length thresholds

### Phase 3: Faceting + Highlights

- Facet counting in WASM (bitmap and column scan)
- `.facets()` API on DataFrame
- `FACET` clause in SQL
- Optional highlight generation (mark matched terms in result snippets)

### Phase 4: Hybrid Fusion

- RRF implementation in `merge.ts`
- Weighted fusion alternative
- Combined explain output for hybrid queries
- Benchmark: hybrid vs BM25-only vs vector-only on relevance benchmarks

### Phase 5: CJK + Extended Tokenizers

- Bigram tokenizer for CJK
- Language detection (Unicode script ranges)
- Stemming (Porter stemmer for English)
- Configurable stopword lists

---

## 12. Dependencies

| Dependency | Status | Notes |
|------------|--------|-------|
| WASM SIMD engine | **Done** | Zig SIMD dispatch already exists |
| Fragment storage on R2 | **Done** | Same R2 path as Lance fragments |
| Page-level caching | **Done** | WASM buffer pool + caches.default |
| Fragment DO fan-out | **Done** | Parallel search across fragments |
| Partial aggregation merge | **Done** | For merging top-K across fragments |
| SQL parser | **Done** | Needs MATCH() added to grammar |
| DataFrame API | **Done** | Needs .search() method added |

No external dependencies. Everything builds on existing infrastructure.

---

## 13. Competitive Positioning

| Feature | QueryMode (target) | Meilisearch | Typesense | Algolia |
|---------|-------------------|-------------|-----------|---------|
| Serverless | Yes (CF Workers) | No (long-running) | No | Managed |
| Edge-deployed | Yes (global) | No (single region) | No | Yes (47 DCs) |
| OLAP + Search | Yes | No | No | No |
| Vector + Search | Yes (hybrid) | Yes | Yes | Yes |
| Pay-per-query | Yes (DO billing) | No (server cost) | No | Yes (expensive) |
| Open source | Yes (MIT) | Yes (MIT/BSL) | GPL-3 | No |
| Index overhead | <2x target | ~20x | RAM-bound | Proprietary |
| Typo tolerance | Yes (Phase 2) | Yes | Yes | Yes |
| Cold query | <500ms | N/A (always warm) | N/A | N/A |
| Warm query | <50ms | <50ms | <50ms | <20ms |
