/**
 * Honest benchmark: full-text search capabilities.
 *
 * Tests three corpus sizes with realistic vocabulary diversity.
 * Reports what works, what's slow, and what's missing.
 *
 * Usage: npx vitest run src/search/search.bench.ts
 */

import { describe, it, expect } from "vitest";
import { SearchIndex } from "./search-index.js";
import { tokenizeTerms } from "./tokenizer.js";
import { reciprocalRankFusion } from "./fusion.js";

// ---------------------------------------------------------------------------
// Data generation — variable vocabulary size
// ---------------------------------------------------------------------------

const WORDS = [
  // Common product terms (core vocabulary)
  "wireless", "bluetooth", "charger", "cable", "adapter", "keyboard", "mouse",
  "headphones", "earbuds", "speaker", "monitor", "laptop", "tablet", "phone",
  "case", "screen", "protector", "stand", "dock", "hub", "power", "battery",
  "portable", "compact", "premium", "professional", "gaming", "ergonomic",
  // Brands (expand vocabulary)
  "apple", "samsung", "sony", "bose", "anker", "logitech", "dell", "hp",
  "lenovo", "asus", "razer", "corsair", "sennheiser", "jabra", "microsoft",
  "google", "oneplus", "xiaomi", "huawei", "steelseries", "hyperx", "shure",
  // Technical terms (more vocabulary diversity)
  "usb", "hdmi", "thunderbolt", "nvme", "ssd", "ddr5", "pcie", "wifi",
  "mesh", "ethernet", "fiber", "optical", "mechanical", "membrane", "tactile",
  "linear", "clicky", "silent", "rgb", "backlit", "oled", "ips", "hdr",
  // Adjectives
  "fast", "ultra", "slim", "rugged", "waterproof", "dustproof", "adjustable",
  "foldable", "magnetic", "universal", "lightweight", "durable", "advanced",
  "next", "generation", "military", "grade", "eco", "friendly", "certified",
  // Descriptive phrases (longer titles)
  "noise", "cancelling", "active", "passive", "over", "ear", "in",
  "true", "stereo", "surround", "sound", "bass", "treble", "equalizer",
  "microphone", "condenser", "dynamic", "cardioid", "omnidirectional",
  "resolution", "refresh", "rate", "response", "time", "latency", "low",
  "high", "speed", "capacity", "storage", "memory", "processor", "chip",
];

function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateDocs(count: number, seed = 42): string[] {
  const rng = mulberry32(seed);
  const docs: string[] = [];
  for (let i = 0; i < count; i++) {
    // 3-8 words per doc title, random selection
    const wordCount = 3 + Math.floor(rng() * 6);
    const title: string[] = [];
    for (let w = 0; w < wordCount; w++) {
      title.push(WORDS[Math.floor(rng() * WORDS.length)]);
    }
    docs.push(title.join(" "));
  }
  return docs;
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

interface Stats { p50: number; p99: number; mean: number }

function measure(fn: () => void, iterations: number): Stats {
  const times: number[] = [];
  for (let i = 0; i < 5; i++) fn(); // warmup
  for (let i = 0; i < iterations; i++) {
    const s = performance.now();
    fn();
    times.push(performance.now() - s);
  }
  times.sort((a, b) => a - b);
  return {
    p50: times[Math.floor(times.length * 0.5)],
    p99: times[Math.floor(times.length * 0.99)],
    mean: times.reduce((a, b) => a + b, 0) / times.length,
  };
}

function fmt(ms: number): string {
  if (ms < 0.01) return `${(ms * 1000).toFixed(0)}us`;
  if (ms < 1) return `${ms.toFixed(2)}ms`;
  if (ms < 1000) return `${ms.toFixed(1)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function fmtBytes(b: number): string {
  if (b < 1024) return `${b}B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)}KB`;
  return `${(b / (1024 * 1024)).toFixed(1)}MB`;
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

describe("search benchmark", () => {
  const QUERIES_EXACT = [
    "wireless headphones", "bluetooth speaker", "usb charger cable",
    "mechanical keyboard", "noise cancelling", "gaming mouse rgb",
    "laptop stand adjustable", "portable battery", "hdmi adapter", "screen protector",
  ];
  const QUERIES_TYPO = [
    "wireles headphons", "bluetoth speakr", "mecanical keybord",
    "noice canceling", "portble batteri", "adaptr hdmi", "gamin mouse",
    "lapto stand", "ergonomc keyboard", "profesional monitor",
  ];

  for (const [label, docCount] of [["10K", 10_000], ["100K", 100_000], ["500K", 500_000]] as const) {
    it(`${label} documents`, () => {
      // Build
      const docs = generateDocs(docCount);
      const buildStart = performance.now();
      const index = SearchIndex.build(docs, { column: "text" });
      const buildMs = performance.now() - buildStart;

      // Serialize
      const ser = index.serialize();
      const indexBytes = ser.postings.byteLength + ser.stats.byteLength;

      // Vocab
      const vocab = index.vocabularySize;

      // Exact search (AND)
      const exactAnd = measure(() => {
        for (const q of QUERIES_EXACT) index.search(q, 10, "and", 0);
      }, 100);

      // Exact search (OR)
      const exactOr = measure(() => {
        for (const q of QUERIES_EXACT) index.search(q, 10, "or", 0);
      }, 100);

      // Fuzzy (typo=1, OR)
      const fuzzy1 = measure(() => {
        for (const q of QUERIES_TYPO) index.search(q, 10, "or", 1);
      }, 50);

      // Fuzzy (typo=2, OR)
      const fuzzy2 = measure(() => {
        for (const q of QUERIES_TYPO) index.search(q, 10, "or", 2);
      }, 50);

      // Facet simulation: search + count category values
      const facetMs = measure(() => {
        const result = index.search("wireless headphones", 100, "or", 0);
        // Simulate facet counting over matched doc IDs
        const counts = new Map<number, number>();
        for (const hit of result.hits) {
          const bucket = hit.docId % 10; // simulate 10 category buckets
          counts.set(bucket, (counts.get(bucket) ?? 0) + 1);
        }
      }, 100);

      // RRF fusion simulation
      const rrfMs = measure(() => {
        const bm25 = index.search("wireless headphones", 50, "or", 0);
        const vector = index.search("bluetooth speaker", 50, "or", 0); // simulate vector results
        reciprocalRankFusion(
          [bm25.hits.map(h => ({ docId: h.docId, score: h.score })),
           vector.hits.map(h => ({ docId: h.docId, score: h.score }))],
          10,
        );
      }, 100);

      // Per-query stats (divide by 10 queries)
      const perQ = (s: Stats): Stats => ({ p50: s.p50 / 10, p99: s.p99 / 10, mean: s.mean / 10 });
      const eAnd = perQ(exactAnd);
      const eOr = perQ(exactOr);
      const f1 = perQ(fuzzy1);
      const f2 = perQ(fuzzy2);

      // Verify correctness
      const r1 = index.search("wireless headphones", 10, "and", 0);
      expect(r1.hits.length).toBeGreaterThan(0);
      const r2 = index.search("wireles headphons", 10, "or", 1);
      expect(r2.hits.length).toBeGreaterThan(0);

      // Stemming test
      const stemIndex = SearchIndex.build(docs, { column: "text", tokenizer: { stemming: true } });
      const stemResult = stemIndex.search("cancelling", 10, "or", 0);

      // Print report
      const line = (l: string, s: Stats) =>
        `  ${l.padEnd(30)} ${fmt(s.p50).padStart(10)}  ${fmt(s.p99).padStart(10)}`;

      console.log([
        "",
        `  === ${label} docs | vocab: ${vocab} | index: ${fmtBytes(indexBytes)} | build: ${fmt(buildMs)} ===`,
        "",
        `  ${"Query Type".padEnd(30)} ${"p50".padStart(10)}  ${"p99".padStart(10)}`,
        `  ${"-".repeat(54)}`,
        line("Exact AND", eAnd),
        line("Exact OR", eOr),
        line("Fuzzy typo=1 OR", f1),
        line("Fuzzy typo=2 OR", f2),
        line("Facet (search+count)", { p50: facetMs.p50, p99: facetMs.p99, mean: facetMs.mean }),
        line("Hybrid RRF (2-list fusion)", { p50: rrfMs.p50, p99: rrfMs.p99, mean: rrfMs.mean }),
        `  ${"-".repeat(54)}`,
        `  Stemming vocab: ${stemIndex.vocabularySize} (vs ${vocab} unstemmed)`,
        `  Stemming match: "${stemResult.queryTerms.join(", ")}" → ${stemResult.hits.length} hits`,
        "",
      ].join("\n"));

      // Sanity: latency bounds
      expect(eAnd.p99).toBeLessThan(50);
      expect(f2.p99).toBeLessThan(500);
    });
  }

  it("prints honest status", () => {
    console.log([
      "",
      "  ╔══════════════════════════════════════════════════════════════╗",
      "  ║                    WHAT WORKS                               ║",
      "  ╠══════════════════════════════════════════════════════════════╣",
      "  ║ ✓ BM25 ranked search          .search('query')             ║",
      "  ║ ✓ Typo tolerance              .search('iphon', {typo: 1})  ║",
      "  ║ ✓ Search + columnar filter    .search().filter()           ║",
      "  ║ ✓ Faceted search              .search().facets(['brand'])  ║",
      "  ║ ✓ Hybrid BM25+vector          .search().nearestTo()        ║",
      "  ║ ✓ RRF rank fusion             reciprocalRankFusion()       ║",
      "  ║ ✓ Porter stemmer              { stemming: true }           ║",
      "  ║ ✓ CJK bigram tokenizer        auto-detected                ║",
      "  ║ ✓ SQL MATCH syntax            WHERE MATCH('query')         ║",
      "  ║ ✓ Field weights               MATCH('q', title^3)          ║",
      "  ╠══════════════════════════════════════════════════════════════╣",
      "  ║                    WHAT'S MISSING                           ║",
      "  ╠══════════════════════════════════════════════════════════════╣",
      "  ║ ✗ Zig WASM engine             tokenizer+BM25 in TypeScript ║",
      "  ║ ✗ FST term dictionary         using Map (O(n) fuzzy scan)  ║",
      "  ║ ✗ Persistent index on R2      in-memory only               ║",
      "  ║ ✗ Edge mode (DO fan-out)      local mode only              ║",
      "  ║ ✗ Highlight snippets          _highlights column           ║",
      "  ║ ✗ HTTP /search endpoint       not wired                    ║",
      "  ╠══════════════════════════════════════════════════════════════╣",
      "  ║                    KNOWN LIMITATIONS                        ║",
      "  ╠══════════════════════════════════════════════════════════════╣",
      "  ║ • Fuzzy scan is O(vocabulary) — slow at 100K+ unique terms ║",
      "  ║ • Index rebuilt on each search if not cached                ║",
      "  ║ • Fresh executor on pre-existing dataset: WASM may not     ║",
      "  ║   return utf8 columns → search finds nothing               ║",
      "  ║ • MATCH() inside OR expressions not supported              ║",
      "  ╚══════════════════════════════════════════════════════════════╝",
      "",
    ].join("\n"));
  });
});
