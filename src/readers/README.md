# readers/ — Ingestion Format Readers

> Read CSV, JSON, and Arrow IPC files into column arrays for fragment building.
> These are ingestion-only formats — query-time formats (Lance, Parquet, Iceberg) live in `src/`.

## Files

| File | Purpose |
|------|---------|
| csv-reader.ts | CSV/TSV parser. Auto-detects delimiter, infers column types (int64/float64/bool/utf8) |
| json-reader.ts | JSON/NDJSON parser. Flattens nested objects via `JSON.stringify`. Type inference |
| arrow-reader.ts | Arrow IPC reader. Reads record batches, validates vtable offsets, extracts typed arrays |

## Shared Patterns

All three readers produce the same output: `FragmentColumn[]` (defined in `wasm-engine.ts`).

```typescript
interface FragmentColumn {
  name: string
  dtype: DataType
  values: Int64Array | Float64Array | string[] | Uint8Array | boolean[]
  nullable?: Uint8Array  // null bitmap
}
```

This feeds directly into `WasmEngine.buildFragment()` → Lance binary → R2.

## Data Flow

```
CSV/JSON/Arrow file
  │
  ▼
reader.ts ──► ReaderRegistry.read(source, format)
  │              Dispatches to correct reader based on format/extension
  │
  ▼
FragmentColumn[] ──► wasm-engine.ts buildFragment()
  │
  ▼
Lance binary ──► R2 storage
```

## Key Details

- **encodeColumnBuffer()** is shared across csv-reader and json-reader (exported from reader.ts) — converts string[] to packed binary with offsets
- **Type inference**: Readers scan values to pick the narrowest type. Priority: int64 > float64 > bool > utf8
- **Nested JSON**: Objects become JSON strings (`JSON.stringify`), not flattened columns
- **Arrow IPC**: Supports Int8–Int64, UInt8–UInt64, Float16–Float64, Utf8, Binary, Bool, FixedSizeList (vectors)
