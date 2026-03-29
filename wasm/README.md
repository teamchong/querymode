# wasm/ — Zig WASM Engine

> 154 files, 90K lines of Zig. Compiles to `src/wasm/querymode.wasm`.
> All compute happens here: SIMD filters, aggregation, hashing, vector search, SQL execution.
> TS never does math — it orchestrates I/O and calls into this engine.
>
> **For WASM bridge invariants, see `specs/invariants.md` section 2.**
> **For byte-level formats, see `specs/data-formats.md`.**

## Build

```bash
cd wasm && zig build -Dtarget=wasm32-wasi
# Output: src/wasm/querymode.wasm
```

## Module Map

```
wasm/src/
├── querymode.zig          Entry point. WASM export table (150+ functions)
├── wasm.zig               WASM-specific glue (memory, exports)
│
├── query/                 ── DuckDB-style Query Engine ──
│   ├── vector_engine.zig  SelectionVector, DataChunk, Vector, LinearHashTable
│   ├── aggregates.zig     Hash aggregation (SUM, COUNT, AVG, MIN, MAX, etc.)
│   ├── executor.zig       Query plan execution
│   ├── ast.zig            Query AST nodes
│   ├── expr.zig           Expression evaluation
│   ├── lexer.zig          Query lexer
│   ├── parser.zig         Query parser
│   └── gpu_*.zig          GPU-accelerated group-by and hash join (experimental)
│
├── sql/                   ── Full SQL Engine ──
│   ├── lexer.zig          SQL tokenizer
│   ├── parser.zig         SQL parser → AST
│   ├── ast.zig            SQL AST types
│   ├── executor.zig       SQL execution engine
│   ├── expr_eval.zig      Expression evaluator (CASE, CAST, arithmetic)
│   ├── where_eval.zig     WHERE clause evaluation
│   ├── group_eval.zig     GROUP BY execution
│   ├── having_eval.zig    HAVING clause filtering
│   ├── window_eval.zig    Window function execution (ROW_NUMBER, RANK, etc.)
│   ├── window_functions.zig  Window function implementations
│   ├── aggregate_functions.zig  Aggregate function implementations
│   ├── scalar_functions.zig     Scalar function implementations
│   ├── set_ops.zig        UNION, INTERSECT, EXCEPT
│   ├── result_ops.zig     Result sorting, limiting, projection
│   ├── result_types.zig   Result type definitions
│   ├── runtime_columns.zig  Runtime column management
│   ├── late_materialization.zig  Streaming batch execution
│   ├── streaming_reader.zig     Streaming data reader
│   ├── codegen/
│   │   ├── fused_codegen.zig    Query plan → single vectorized kernel
│   │   └── simd_ops.zig        SIMD operation primitives for codegen
│   └── planner/
│       ├── planner.zig    Query optimizer / planner
│       └── plan_nodes.zig Plan node types
│
├── wasm/                  ── WASM Runtime Layer ──
│   ├── memory.zig         Heap allocator, resetHeap()
│   ├── buffer_pool.zig    LRU page cache (1024 entries, 64MB)
│   ├── filter_simd.zig    WASM SIMD128 filter kernels (Vec2i64/Vec2f64)
│   ├── simd_search.zig    SIMD dot/L2/cosine for vector search
│   ├── aggregates.zig     WASM-specific aggregate bindings
│   ├── sql_executor.zig   WASM SQL execution entry point
│   ├── column_meta.zig    Column metadata management in WASM
│   ├── string_column.zig  String column encoding/decoding
│   ├── vector_column.zig  Vector column handling
│   ├── fragment_reader.zig Fragment reading in WASM
│   ├── lance_writer.zig   Lance format writing from WASM
│   ├── dataset_writer.zig Dataset writing
│   ├── compression.zig    Compression codecs
│   ├── format.zig         Format detection
│   ├── ivf_pq.zig         IVF-PQ vector index
│   ├── opfs.zig           Origin Private File System (browser)
│   └── *_model.zig        Embedding models (CLIP, MiniLM, TinyBERT)
│
├── format/                ── File Format Parsers ──
│   ├── lance_file.zig     Lance v1/v2 reader
│   ├── lazy_lance_file.zig Lazy Lance reader (demand-paged)
│   ├── parquet_file.zig   Parquet reader
│   ├── parquet_metadata.zig Parquet metadata parser
│   ├── footer.zig         Footer parsing (Lance, Parquet)
│   ├── manifest.zig       Lance manifest reader
│   ├── manifest_writer.zig Lance manifest writer
│   ├── version.zig        Version tracking
│   ├── page_row_index.zig Page-to-row mapping
│   └── format.zig         Format detection and dispatch
│
├── encoding/              ── Codec Layer ──
│   ├── plain.zig          Plain encoding
│   ├── snappy.zig         Snappy decompression
│   ├── encoding.zig       Encoding dispatch
│   ├── csv.zig            CSV codec
│   ├── json.zig           JSON codec
│   ├── arrow_ipc.zig      Arrow IPC codec
│   ├── avro.zig           Avro codec
│   ├── delta.zig          Delta Lake codec
│   ├── iceberg.zig        Iceberg codec
│   ├── orc/               ORC codec
│   ├── parquet/           Parquet codec details
│   ├── writer.zig         Encoding writer
│   └── xlsx.zig           Excel codec
│
├── io/                    ── I/O Abstraction ──
│   ├── reader.zig         Reader trait
│   ├── file_reader.zig    Local file I/O
│   ├── http_reader.zig    HTTP range requests
│   ├── memory_reader.zig  In-memory buffer
│   ├── mmap_reader.zig    Memory-mapped I/O
│   ├── batch_reader.zig   Batched reading
│   ├── s3_client.zig      S3-compatible object storage
│   └── io.zig             I/O dispatch
│
├── embedding/             ── Vector Embeddings ──
│   ├── embedding.zig      Embedding engine dispatch
│   ├── tokenizer.zig      Text tokenizer for embedding models
│   ├── flat_index.zig     Flat (brute-force) vector index
│   ├── ivf_pq_index.zig   IVF-PQ index (partitioned quantized)
│   ├── clip.zig           CLIP model inference
│   ├── minilm.zig         MiniLM model inference
│   ├── onnx.zig           ONNX runtime bindings
│   └── session.zig        Model session management
│
├── proto/                 ── Protobuf/Thrift ──
│   ├── decoder.zig        Protobuf decoder
│   ├── encoder.zig        Protobuf encoder
│   ├── lance_messages.zig Lance-specific protobuf messages
│   ├── schema.zig         Schema protobuf
│   └── thrift.zig         Thrift decoder (for Parquet)
│
├── cli/                   ── CLI Tool ──
│   ├── serve.zig          HTTP server mode
│   ├── query_utils.zig    Query execution helpers
│   ├── ingest/            Ingestion commands
│   ├── enrich/            Data enrichment commands
│   └── ...                Args, output, benchmarking
│
├── ai/                    ── AI/LLM Integration ──
│   ├── gguf.zig           GGUF model format reader
│   └── tinybert.zig       TinyBERT inference
│
├── gpu/                   ── GPU Compute (experimental) ──
│   ├── gpu_context.zig    GPU device management
│   ├── batch_ops.zig      Batched GPU operations
│   ├── hash_table.zig     GPU hash table
│   └── vector_search.zig  GPU vector search
│
├── ── Top-Level Files ──
├── simd.zig               SIMD dispatch: scalar(<64) / SIMD(<20K) / parallel
│                          Vec4F64, Vec8F32
├── columnar_ops.zig       8-wide SIMD columnar filters (Vec8i64/Vec8f64)
├── hash.zig               FNV-1a + Murmur3 finalizer + combineHash
├── table.zig              Table abstraction (column store)
├── value.zig              Value type (tagged union for all data types)
├── dataframe.zig          DataFrame operations
├── dataset.zig            Dataset (multi-fragment) management
├── dataset_writer.zig     Dataset writing
├── *_table.zig            Format-specific table readers (parquet, arrow, etc.)
└── nodejs.zig / python.zig  Language bindings
```

## Key Architecture Patterns

### SIMD Dispatch (simd.zig)
Three tiers based on data size:
- **< 64 elements**: Scalar loop (no SIMD overhead)
- **< 20K elements**: WASM SIMD128 (Vec2i64/Vec2f64 — 128-bit lanes)
- **≥ 20K elements**: Parallel SIMD (chunked across threads)

### DuckDB-Style Internals (query/vector_engine.zig)
- **SelectionVector**: Indices into a column, avoids copying filtered rows
- **DataChunk**: Fixed-size batch of vectors (like DuckDB's 2048-row chunks)
- **Vector**: Single typed column within a chunk
- **LinearHashTable**: Open-addressing hash table for GROUP BY / JOIN

### Fused Codegen (sql/codegen/fused_codegen.zig)
Compiles a query plan into a single vectorized kernel. Instead of operator-per-operator execution, the whole WHERE→AGG→PROJECT pipeline becomes one tight loop. Eliminates per-row virtual dispatch.

### Late Materialization (sql/late_materialization.zig)
Only materializes columns that survive filtering. Filter columns evaluated first; projection columns fetched only for passing rows. Streaming batch execution — never holds full result in memory.

## WASM Export Convention

All WASM exports use byte addresses (not shifted pointers). The TS bridge in `wasm-engine.ts` writes data to the WASM heap and passes byte offsets. Example:

```
TS: const ptr = wasm.alloc(size)    // byte address
    new Float64Array(heap, ptr, n)  // write data
    wasm.someExport(ptr, n)         // pass byte address
```

**Never** divide pointers by element size when calling WASM exports.
