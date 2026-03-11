# Design: QueryMode as Agentic Infrastructure

Status: **Draft — iterating**

## Problem

The agentic workflow has three layers:

```
Source (internet) → Preprocess (to R2) → Query (R2 to agent) → Context Manage (context window)
```

QueryMode covers Query. Source→Preprocess and Context Management are open gaps. Nobody does either well:

- **Source→Preprocess**: Google indexes everything but their quality signals have been gamed for 20 years. AI-generated, SEO-optimized content scores high on PageRank but carries no information. Gemini inherits this — brilliant model, garbage source.
- **Context Management**: RAG is one-shot retrieval (embed → top-k → stuff). Misses structure, can't follow references, no progressive refinement. Hybrid search (BM25 + vector) just doubles down on the wrong model. Special formats (TOON, etc.) optimize the container but not the selection.

## Thesis

QueryMode's composable operator model already solves columnar data problems at scale. Documents are rows. Quality, freshness, embedding, topic — these are columns. Filter, dedup, aggregate, project — these are operators.

Extend QueryMode with domain-specific operators to cover all three layers. Same engine, same API, same infrastructure.

## Non-goals

- Not building a search engine (no web-scale crawl)
- Not building a vector database (existing vector search is sufficient)
- Not replacing the LLM's reasoning (operators prepare context, model reasons over it)
- Not building a framework (operators are building blocks, not a pipeline DSL)

## Layer 1: Source → Preprocess

### Input

CF Browser Rendering API / Crawler API produces markdown from web pages. The raw crawl is a stream of `{ url, domain, crawl_date, markdown, http_status, content_type }` rows.

### Pipeline

```
Raw crawl → Fingerprint → Quality score → Dedup → Embed → Topic cluster → Partition → R2
```

Each stage is a QueryMode operator or Pipeline stage.

### New operators

#### `SimHashOperator`

Near-duplicate detection via SimHash fingerprinting.

```typescript
// Input: rows with a text column
// Output: rows + simhash column (64-bit fingerprint)
.pipe(upstream => new SimHashOperator(upstream, { column: "markdown" }))
```

How it works:
1. Tokenize text into shingles (n-grams of words)
2. Hash each shingle
3. Combine into 64-bit fingerprint where Hamming distance ≈ content similarity
4. Documents within Hamming distance ≤ 3 are near-duplicates

Use case: Catch copy-paste content, syndicated articles, scraped-and-rewritten pages.

Complexity: O(n) per document, O(1) comparison. No embeddings needed.

#### `QualityScoreOperator`

Information density scoring. Not popularity (PageRank), not engagement (clicks) — structural quality of the content itself.

```typescript
.pipe(upstream => new QualityScoreOperator(upstream, { column: "markdown" }))
```

Signals (all computable without an LLM):
- **Code blocks**: presence and ratio to prose (technical content signal)
- **Citations/links**: external references (sourced vs. unsourced)
- **Specificity**: named entities, numbers, dates vs. vague language
- **Structure**: headings, lists, tables (organized vs. wall of text)
- **Information density**: unique tokens / total tokens (high = diverse vocabulary = likely informative)
- **Freshness**: crawl date, published date if extractable

Output: `quality_score` float [0, 1] column.

Open question: Should this be a single score or multiple columns (code_density, citation_count, specificity_score, etc.)? Multiple columns let downstream queries filter on specific dimensions.

#### `TopicClusterOperator`

Semantic grouping via embedding clustering.

```typescript
.pipe(upstream => new TopicClusterOperator(upstream, {
  embeddingColumn: "embedding",
  method: "kmeans",  // or "hdbscan"
  k: 100,            // target cluster count (kmeans only)
}))
```

Output: `topic_cluster` column (integer cluster ID) + optional `topic_label` (derived from cluster centroid's nearest document title).

Use case: Partition the knowledge base by topic for O(1) lookup via partition catalog. "Everything about WebSockets" = one partition.

Open question: K-means needs a fixed k. HDBSCAN finds natural clusters but is O(n log n). For a knowledge base of 10K-100K documents, either works. For 1M+, we'd need an approximate method.

#### `EmbedOperator`

Generate embeddings for text content.

```typescript
.pipe(upstream => new EmbedOperator(upstream, {
  column: "markdown",
  model: "voyage-3",  // or any embedding API
  batchSize: 100,
}))
```

Output: `embedding` float32 vector column.

Implementation note: This operator calls an external API (Voyage, OpenAI, Cohere). It should batch requests, handle rate limits, and cache results. The embedding is written to the row — subsequent queries use QueryMode's existing vector search.

Open question: Where does the embedding API call happen? Options:
1. In the Pipeline stage (simple, but ties the pipeline to an API)
2. As a separate ingest step (embed offline, store in R2, pipeline reads pre-embedded data)
3. Lazy embedding (embed on first query, cache in column)

### Dedup strategy

Three levels, applied in order (cheapest first):

1. **URL dedup** — exact URL match, keep latest crawl. O(1) hash lookup.
2. **SimHash dedup** — Hamming distance ≤ 3, keep highest quality score. O(n) scan, O(1) comparison.
3. **Embedding cluster dedup** — within each topic cluster, if documents are > 0.95 cosine similarity, keep highest quality. O(n²) within cluster, but clusters are small.

### Output schema

```
knowledge_base (partitioned by topic_cluster)
├── url          : utf8
├── domain       : utf8
├── title        : utf8
├── crawl_date   : int64 (epoch ms)
├── quality_score: float64
├── simhash      : int64
├── topic_cluster: int32
├── embedding    : float32[dim]
├── markdown     : utf8
├── sections     : utf8[]  (split by heading)
└── section_headings : utf8[]
```

The `sections` and `section_headings` columns enable section-level reads without loading the full document.

## Layer 2: Context Management as Code

### The model

Agents navigate large codebases using: Glob (find files by pattern), Grep (search content), Read (load specific lines). This works because code has structure — files, directories, imports, functions.

Apply the same model to a knowledge base:

| Code navigation | Knowledge navigation |
|---|---|
| `Glob("**/*.ts")` | `filter("topic", "eq", "websockets")` |
| `Grep("handleAuth")` | `filter("markdown", "like", "%authentication%")` or vector search |
| `Read(file, 100, 50)` | `filter("doc_id", "eq", url).select("sections[2]")` |
| Follow imports | Follow references/citations between documents |
| Run tests | Check quality score, cross-reference claims |

### New operators

#### `TokenBudgetOperator`

Like LIMIT but for tokens. Fits results into a context window budget.

```typescript
.pipe(upstream => new TokenBudgetOperator(upstream, {
  maxTokens: 8000,
  contentColumn: "content",
  priority: "relevance_score",  // which rows to keep when budget exhausted
  tokenizer: "cl100k",          // or simple word-based approximation
}))
```

How it works:
1. Pull rows from upstream in priority order
2. Count tokens per row (exact or approximate)
3. Emit rows until budget exhausted
4. Optionally truncate the last row to fit exactly

Open question: Exact tokenization (tiktoken/cl100k) is slow and adds a dependency. A word-based approximation (words × 1.3) is fast and usually within 10% accuracy. Start with approximation, add exact as option?

#### `SectionExtractOperator`

Extract relevant sections from documents. Like `Read(file, offset, limit)` for documents.

```typescript
.pipe(upstream => new SectionExtractOperator(upstream, {
  query: "How does WebSocket upgrade work?",
  queryEmbedding: queryVec,
  maxSectionsPerDoc: 3,
  sectionColumn: "sections",
  headingColumn: "section_headings",
}))
```

How it works:
1. For each document row, iterate sections
2. Score each section against the query (embedding similarity or keyword match)
3. Keep top-k sections per document
4. Replace `markdown` column with concatenated relevant sections

This is the key operator that prevents "load the whole document into context." Most documents have 1-2 relevant sections out of 20+.

#### `RedundancyFilterOperator`

Remove duplicate information across selected documents/sections.

```typescript
.pipe(upstream => new RedundancyFilterOperator(upstream, {
  contentColumn: "content",
  threshold: 0.85,  // cosine similarity threshold for "redundant"
}))
```

How it works:
1. Pull all rows, compute pairwise similarity on content embeddings
2. Greedy selection: take highest quality row, remove all rows within threshold
3. Repeat until no more redundant pairs

Use case: 5 documents all explain "how HTTP/2 multiplexing works." Keep the best one, drop the rest. The agent gets diverse information, not 5 versions of the same explanation.

Open question: Pairwise similarity is O(n²). For small result sets (10-50 sections after TokenBudget), this is fine. For larger sets, use approximate methods (LSH).

### Context query patterns

The agent uses DataFrame queries to navigate the knowledge base, not a RAG pipeline:

```typescript
// Pattern 1: Topic exploration
const overview = await qm.table("knowledge")
  .filter("topic_cluster", "eq", topicId)
  .filter("quality_score", "gt", 0.5)
  .sort("quality_score", "desc")
  .select("url", "title", "section_headings")
  .limit(20)
  .collect()

// Pattern 2: Targeted search
const results = await qm.table("knowledge")
  .vector("embedding", queryVec, 30)
  .filter("quality_score", "gt", 0.3)
  .pipe(upstream => new SectionExtractOperator(upstream, { query, queryEmbedding }))
  .pipe(upstream => new RedundancyFilterOperator(upstream, { threshold: 0.85 }))
  .pipe(upstream => new TokenBudgetOperator(upstream, { maxTokens: 8000 }))
  .collect()

// Pattern 3: Cross-reference (follow citations)
const sources = await qm.table("knowledge")
  .filter("url", "in", citedUrls)
  .pipe(upstream => new SectionExtractOperator(upstream, { query: claim }))
  .collect()

// Pattern 4: Freshness-aware (decay old information)
const current = await qm.table("knowledge")
  .filter("topic_cluster", "eq", topicId)
  .pipe(upstream => new FreshnessDecayOperator(upstream, {
    dateColumn: "crawl_date",
    halfLifeDays: 90,  // score halves every 90 days
  }))
  .sort("decayed_score", "desc")
  .limit(10)
  .collect()
```

### How this differs from RAG

| | RAG | Context as Code |
|---|---|---|
| Selection | One-shot top-k | Iterative, agent-directed |
| Granularity | Fixed chunk size | Section-level, variable |
| Quality signal | None (or reranker) | Quality score column, filterable |
| Dedup | None | RedundancyFilterOperator |
| Budget | Hope it fits | TokenBudgetOperator (exact) |
| Structure | Flat chunks | Documents → sections → headings |
| Follow-up | Re-embed, re-retrieve | Navigate: filter, read, follow refs |
| Freshness | None | FreshnessDecayOperator |

## Implementation plan

### Phase 1: Ingest operators

Build the preprocessing pipeline operators:

1. `SimHashOperator` — pure TypeScript, no external deps
2. `QualityScoreOperator` — pure TypeScript, heuristic scoring
3. `SectionSplitOperator` — split markdown by headings into sections array

These are enough to crawl → dedup → quality score → store. Embedding and clustering can use existing vector search + groupBy.

### Phase 2: Context operators

Build the context management operators:

4. `TokenBudgetOperator` — word-based approximation, exact tokenizer optional
5. `SectionExtractOperator` — section-level relevance scoring
6. `RedundancyFilterOperator` — greedy dedup on content similarity

### Phase 3: Agent integration

Build the agent-facing API that uses these operators:

7. `KnowledgeBase` class — wraps QueryMode + context operators into a simple interface
8. Agent tool definitions — `search()`, `read()`, `explore()`, `verify()`

### Phase 4: CF Crawler integration

9. Worker that calls CF Browser Rendering → feeds into ingest pipeline
10. Scheduled crawl + incremental update (only re-crawl stale URLs)

## Open questions

1. **Where do embeddings come from?** External API (Voyage, OpenAI) adds latency and cost. Could use a small local model (ONNX in WASM?) for ingest-time embedding, but that's a big scope expansion.

2. **Section granularity**: Split by H1? H2? H3? Paragraphs? Probably configurable, default to H2 (major sections).

3. **Quality score calibration**: The heuristic signals need tuning. Start with equal weights, calibrate against a labeled set of "good" vs. "garbage" documents.

4. **Knowledge base size target**: 10K documents (focused domain) vs. 1M documents (broad knowledge)? This affects clustering method and partition strategy.

5. **Update strategy**: Full re-crawl vs. incremental? Incremental needs change detection (compare simhash of new crawl vs. stored). Incremental is harder but necessary at scale.

6. **Token counting**: Approximate (words × 1.3) vs. exact (tiktoken)? Approximate is fast and dependency-free. Exact requires a tokenizer library or WASM module.

7. **Should `KnowledgeBase` be a separate package?** Or a module within querymode? Keeping it in querymode means one dependency for the agent developer. Separate package means querymode stays focused on query.
