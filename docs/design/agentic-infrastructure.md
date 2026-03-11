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

## Core insight: iteration

The main reason code navigation works for agents is that **you can iterate on it**. You can't iterate on RAG.

RAG is a black box: embed → retrieve → stuff → hope. If the result is wrong, you can't:
- Diff what changed between two retrievals
- Version control your retrieval strategy
- Write a test that asserts "this query should return these documents"
- Gradually refine the result by narrowing the search

Code navigation is iterative: Glob → Grep → Read → refine → Read more. Each step is observable, testable, version-controllable. The agent sees what it got, decides if it's enough, and adjusts.

**Context management as code** means the same thing: the context selection logic is code you can diff, test, version, and iterate on. Not a pipeline you configure and pray.

## Landscape analysis

### Layer 1: Crawl → Preprocess (existing tools)

| Tool | What it does | Gap |
|---|---|---|
| [CF /crawl endpoint](https://developers.cloudflare.com/browser-rendering/rest-api/crawl-endpoint/) (March 2026) | Crawl entire sites, return markdown. Async job, respects robots.txt | No quality filtering, no dedup |
| [CF Markdown for Agents](https://blog.cloudflare.com/markdown-for-agents/) | Any CF-proxied URL returns markdown via `Accept: text/markdown` | Single-page, no pipeline |
| [Crawl4AI](https://github.com/unclecode/crawl4ai) (58k+ stars) | Open-source, LLM-ready markdown, adaptive crawling, local-first | Python-only, no quality scoring built-in |
| [Firecrawl](https://www.firecrawl.dev/) | Managed crawl + markdown, LangChain integration | SaaS, data goes through their servers |
| [Jina Reader](https://r.jina.ai/) | Prepend URL, get markdown. Zero setup | Rate limited, single-page, no pipeline |
| [FineWeb-Edu](https://arxiv.org/html/2406.17557v1) (HuggingFace) | Gold standard: MinHash dedup + LLM quality classifier → 1.3T tokens | Training data pipeline, not agent knowledge base |
| [LSHBloom](https://arxiv.org/html/2411.04257v3) | Drop-in MinHash replacement, better scaling | Research, no production tool |
| [Craw4LLM](https://arxiv.org/abs/2502.13347) | 21% of URLs → same LLM performance. Quality-guided crawling | Research paper, not a tool |

**Key learning from FineWeb**: MinHash dedup with 5-grams, 112 hash functions, 14 buckets of 8. Per-snapshot dedup outperformed global dedup. Quality classifier trained on LLM annotations (Llama-3-70B scored 500K samples on 0-5 scale). Threshold of 3 gave best results. The classifier is a frozen Snowflake-arctic-embed with a regression head — 82% F1.

**Key learning from Craw4LLM**: You don't need to crawl everything. Quality-guided crawling (score pages, follow links from high-scoring pages) gets 79% fewer URLs for the same model performance. This is relevant for agent knowledge bases — small and high-quality beats large and noisy.

**What's missing**: Nobody connects crawl → quality → dedup → store in a single composable pipeline. Crawl4AI crawls. FineWeb filters. LlamaIndex indexes. Three separate tools with no composition.

### Layer 2: Context Management (existing approaches)

| Approach | What it does | Gap |
|---|---|---|
| [RAG (traditional)](https://arxiv.org/abs/2501.09136) | Embed → top-k → stuff into context | One-shot, no iteration, no quality signal |
| [Agentic RAG / A-RAG](https://arxiv.org/html/2602.03442v1) | Agent controls retrieval tools (lexical + semantic + chunk-level) | Better, but still retrieval-centric |
| [ACE framework](https://arxiv.org/abs/2510.04618) | Treats context as evolving playbooks with generation/reflection/curation | Academic, +10.6% on benchmarks |
| [LlamaIndex](https://www.llamaindex.ai/) | Hierarchical indices, knowledge graphs, query decomposition | Over-abstracted, "magic" internals |
| [LangChain/LangGraph](https://docs.langchain.com/oss/python/langchain/context-engineering) | Context flows as code, middleware pattern | Abstraction overload, version instability |
| [Harness Engineering](https://openai.com/index/harness-engineering/) (OpenAI, Feb 2026) | AGENTS.md as table of contents → progressive disclosure | Pattern, not a tool |
| [Context Engineering](https://martinfowler.com/articles/exploring-gen-ai/context-engineering-coding-agents.html) (Fowler, Jan 2026) | Layered context, skills for lazy-loading | Describes the problem well, no unified solution |
| [MCP](https://modelcontextprotocol.io/) (Anthropic → Linux Foundation) | Universal protocol for agent ↔ tool communication | Transport layer, not context selection |

**Key findings**:
- [Karpathy's analogy](https://weaviate.io/blog/context-engineering): LLM = CPU, context window = RAM. Context engineering = OS deciding what fits in RAM.
- [OpenAI's production data](https://openai.com/index/harness-engineering/): past ~60% context window utilization, more context makes agents actively worse.
- [ETH Zurich study](https://www.infoq.com/news/2026/03/agents-context-file-value-review/): AGENTS.md files can actually hinder agents. LLM-generated context files have marginal negative effect.
- [RAGFlow review](https://ragflow.io/blog/rag-review-2025-from-rag-to-context): RAG is evolving from "retrieval-augmented generation" into "context engine."
- [LangChain state of agents](https://www.langchain.com/state-of-agent-engineering): 57.3% of respondents have agents in production. Hallucinations and context management are top challenges at scale.

**What's missing**: Everyone treats context selection as a retrieval problem (find similar things). Nobody treats it as a **code navigation problem** (explore, search, read, refine, iterate). The agent tools that work for codebases — Glob, Grep, Read with offset/limit — are the right model, but nobody has applied them to knowledge bases.

### The composition gap

The crawl tools (Crawl4AI, Firecrawl, CF /crawl) produce markdown. The retrieval tools (LlamaIndex, LangChain) consume indexed chunks. The context tools (harness engineering, MCP) manage what enters the window. These are three separate ecosystems.

**Nobody has a single composable system where**:
1. Crawl output flows through quality/dedup operators
2. The same operator model handles retrieval (filter, vector search, sort)
3. The same operator model handles context selection (token budget, section extraction, redundancy filtering)
4. Every step is code you can diff, test, and iterate on

QueryMode's operator pipeline is uniquely positioned here.

## Thesis

QueryMode's composable operator model already solves columnar data problems at scale. Documents are rows. Quality, freshness, embedding, topic — these are columns. Filter, dedup, aggregate, project — these are operators.

Extend QueryMode with domain-specific operators to cover all three layers. Same engine, same API, same infrastructure. The key property: **every step is code** — testable, versionable, iterable.

## Non-goals

- Not building a search engine (no web-scale crawl)
- Not building a vector database (existing vector search is sufficient)
- Not replacing the LLM's reasoning (operators prepare context, model reasons over it)
- Not building a framework (operators are building blocks, not a pipeline DSL)
- Not competing with LlamaIndex/LangChain on orchestration (they orchestrate, we compute)

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

Near-duplicate detection via SimHash fingerprinting. Based on the [MinHash approach used in FineWeb](https://arxiv.org/html/2406.17557v1) (5-gram shingles, 112 hash functions, 14 buckets of 8), but using SimHash for single-fingerprint storage efficiency.

```typescript
// Input: rows with a text column
// Output: rows + simhash column (64-bit fingerprint)
.pipe(upstream => new SimHashOperator(upstream, { column: "markdown" }))
```

How it works:
1. Tokenize text into shingles (n-grams of words, default n=5 per FineWeb)
2. Hash each shingle
3. Combine into 64-bit fingerprint where Hamming distance ≈ content similarity
4. Documents within Hamming distance ≤ 3 are near-duplicates

Use case: Catch copy-paste content, syndicated articles, scraped-and-rewritten pages.

Complexity: O(n) per document, O(1) comparison. No embeddings needed.

Note: FineWeb found per-snapshot dedup outperformed global dedup. For knowledge bases with incremental crawls, the same applies — dedup within each crawl batch, not across all time. [LSHBloom](https://arxiv.org/html/2411.04257v3) offers a more scalable drop-in replacement if the corpus exceeds 10M documents.

#### `QualityScoreOperator`

Information density scoring. Not popularity (PageRank), not engagement (clicks) — structural quality of the content itself. Inspired by [FineWeb-Edu's quality classifier](https://arxiv.org/html/2406.17557v1) (LLM-scored training data → frozen Snowflake-arctic-embed + regression head, 82% F1), but using heuristic signals that run without an LLM call.

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

FineWeb-Edu used a 0-5 LLM annotation scale with a threshold of 3 for "educational quality." Our heuristic approach is coarser but runs at crawl speed with zero API cost. For higher accuracy, an optional `model` parameter could call [CF Workers AI](https://developers.cloudflare.com/workers-ai/) for LLM-based scoring at the edge.

Open question: Should this be a single score or multiple columns (code_density, citation_count, specificity_score, etc.)? Multiple columns let downstream queries filter on specific dimensions. [Craw4LLM](https://arxiv.org/abs/2502.13347) found that quality-guided crawling (21% of URLs → same performance) benefits from multi-dimensional quality signals.

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

### How this differs from existing approaches

| | Traditional RAG | LlamaIndex | LangChain | A-RAG | Context as Code (QueryMode) |
|---|---|---|---|---|---|
| Selection | One-shot top-k | Hierarchical index routing | Middleware chain | Agent-controlled retrieval tools | Iterative DataFrame queries |
| Granularity | Fixed chunk size | Node-level | Document/chunk | Chunk + rerank | Section-level, variable |
| Quality signal | None (or reranker) | None | None | Reranker | Quality score column, filterable |
| Dedup | None | None | None | None | RedundancyFilterOperator |
| Budget | Hope it fits | None | Token counting middleware | None | TokenBudgetOperator (exact) |
| Structure | Flat chunks | Tree of nodes | Flat documents | Flat chunks | Documents → sections → headings |
| Follow-up | Re-embed, re-retrieve | Re-traverse tree | Re-run chain | Agent decides next retrieval | Navigate: filter, read, follow refs |
| Freshness | None | None | None | None | FreshnessDecayOperator |
| Iteration | No | Limited | Chain re-run | Yes (agent loop) | Yes (code you can diff/test/version) |
| Composability | Fixed pipeline | Plugin-based | Middleware chain | Tool-based | Operator pipeline (same as query) |

Key insight from [OpenAI's Harness Engineering](https://openai.com/index/harness-engineering/): past ~60% context utilization, agents get worse. TokenBudgetOperator enforces this directly. [A-RAG](https://arxiv.org/html/2602.03442v1) lets the agent control retrieval but still treats context as "stuff it in." [ACE](https://arxiv.org/abs/2510.04618) treats context as evolving playbooks — closest to our model, but academic (+10.6% on benchmarks, no production tool).

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

1. **Where do embeddings come from?** External API (Voyage, OpenAI) adds latency and cost. [CF Workers AI](https://developers.cloudflare.com/workers-ai/) runs embedding models at the edge (bge-base, bge-large) with no external API call. Could also compile a small model to WASM for fully self-contained embedding, but that's a big scope expansion.

2. **Section granularity**: Split by H1? H2? H3? Paragraphs? Probably configurable, default to H2 (major sections).

3. **Quality score calibration**: The heuristic signals need tuning. Start with equal weights, calibrate against a labeled set of "good" vs. "garbage" documents. FineWeb-Edu's approach (LLM-score 500K samples, train classifier) could be a Phase 2 refinement.

4. **Knowledge base size target**: 10K documents (focused domain) vs. 1M documents (broad knowledge)? This affects clustering method and partition strategy.

5. **Update strategy**: Full re-crawl vs. incremental? Incremental needs change detection (compare simhash of new crawl vs. stored). Incremental is harder but necessary at scale.

6. **Token counting**: Approximate (words × 1.3) vs. exact (tiktoken)? Approximate is fast and dependency-free. A WASM token counter (Zig, reusing the existing WASM build pipeline) would be both fast and exact — worth building if token budget accuracy matters.

7. **Should `KnowledgeBase` be a separate package?** Or a module within querymode? Keeping it in querymode means one dependency for the agent developer. Separate package means querymode stays focused on query.

8. **MCP server integration**: Should QueryMode expose knowledge base tools via [MCP](https://modelcontextprotocol.io/)? This would let any MCP-compatible agent (Claude, etc.) use QueryMode as a knowledge source without custom integration. The protocol is now under the Linux Foundation — likely to become the standard.

9. **Agent tool granularity**: [Harness Engineering](https://openai.com/index/harness-engineering/) recommends tools that are "table of contents, not encyclopedia." The knowledge base tools (search, read, explore, verify) should return pointers and summaries first, full content only on explicit read. This matches the Glob→Grep→Read pattern.

10. **Crawl quality feedback loop**: [Craw4LLM](https://arxiv.org/abs/2502.13347) showed quality-guided crawling reduces URLs by 79%. Should the crawler use quality scores from previous crawls to prioritize which links to follow next? This creates a feedback loop: crawl → score → crawl better.

## References

1. **FineWeb / FineWeb-Edu** — Penedo et al. (2024). "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale." https://arxiv.org/html/2406.17557v1
2. **LSHBloom** — Jennings & Niekum (2024). "LSHBloom: Memory-efficient, Extreme-scale Document Deduplication." https://arxiv.org/html/2411.04257v3
3. **Craw4LLM** — Cao et al. (2025). "Crawl Less, Generate More: Efficient Web Crawling for LLM Pre-Training." https://arxiv.org/abs/2502.13347
4. **Agentic RAG (A-RAG)** — Feng et al. (2026). "A-RAG: Agentic Retrieval-Augmented Generation." https://arxiv.org/html/2602.03442v1
5. **ACE Framework** — Nair et al. (2025). "Agentic Context Engineering." https://arxiv.org/abs/2510.04618
6. **Harness Engineering** — OpenAI (Feb 2026). "How OpenAI builds software agents." https://openai.com/index/harness-engineering/
7. **Context Engineering (Fowler)** — Martin Fowler (Jan 2026). "Context Engineering for Coding Agents." https://martinfowler.com/articles/exploring-gen-ai/context-engineering-coding-agents.html
8. **Karpathy on Context Engineering** — Weaviate blog. https://weaviate.io/blog/context-engineering
9. **RAGFlow review** — "RAG Review 2025: From RAG to Context." https://ragflow.io/blog/rag-review-2025-from-rag-to-context
10. **LangChain State of Agents** — https://www.langchain.com/state-of-agent-engineering
11. **ETH Zurich AGENTS.md study** — InfoQ (March 2026). https://www.infoq.com/news/2026/03/agents-context-file-value-review/
12. **CF Markdown for Agents** — Cloudflare blog. https://blog.cloudflare.com/markdown-for-agents/
13. **CF /crawl endpoint** — Cloudflare docs. https://developers.cloudflare.com/browser-rendering/rest-api/crawl-endpoint/
14. **CF Workers AI** — https://developers.cloudflare.com/workers-ai/
15. **MCP (Model Context Protocol)** — https://modelcontextprotocol.io/
16. **Crawl4AI** — Open-source LLM-ready crawler. https://github.com/unclecode/crawl4ai
17. **Firecrawl** — Managed crawl-to-markdown. https://www.firecrawl.dev/
18. **Jina Reader** — Zero-setup URL-to-markdown. https://r.jina.ai/
19. **RAG survey** — Fan et al. (2025). "A Survey on RAG." https://arxiv.org/abs/2501.09136
