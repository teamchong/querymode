# Spec: Data Formats

Status: **Implemented** — byte-level documentation of all wire and storage formats.

---

## 1. QMCB (QueryMode Columnar Binary)

The internal columnar wire format. Used for DO-to-DO transfer and WASM result exchange.

### Layout

```
Offset  Size     Field
──────  ───────  ────────────────────
0       4        Magic: 0x42434D51 ("QMCB" LE)
4       4        Row count (uint32 LE)
8       4        Column count (uint32 LE)
12      ...      Column headers (repeated × column count)
...     ...      Column data (concatenated)
```

### Column Header (per column)

```
Offset  Size     Field
──────  ───────  ─────────────────
0       4        Name length (uint32 LE)
4       N        Name (UTF-8 bytes)
4+N     1        Dtype tag
5+N     4        Data byte length (uint32 LE)
9+N     1        Has null bitmap (0 or 1)
[10+N   4]       Null bitmap byte length (only if has null bitmap = 1)
```

### Dtype Tags (stable — part of wire format)

| Tag | Type | Element Size | Notes |
|-----|------|-------------|-------|
| 0 | float64 | 8 bytes | |
| 1 | int64 | 8 bytes | |
| 2 | utf8 | variable | Data = offsets (uint32 × (rowCount+1)) + string bytes |
| 3 | bool | 1 byte | One byte per value (not bit-packed) |
| 4 | f32vec | variable | Preceded by 4-byte dimension (uint32 LE), then float32 × dim × rowCount |
| 5 | null | 0 bytes | All values null (column carries no data) |
| 6 | int32 | 4 bytes | |
| 7 | float32 | 4 bytes | |

### UTF-8 Column Data Layout

```
[offsets: uint32 LE × (rowCount + 1)] [string bytes]

offsets[0] = 0 (always)
offsets[i] - offsets[i-1] = byte length of string i-1
String i = bytes[offsets[i]..offsets[i+1]]
```

### Null Bitmap

```
Bit-packed, LSB first. Bit set = null.
Byte count = ceil(rowCount / 8)
```

---

## 2. Lance Fragment (single file)

Binary columnar file. Written by `WasmEngine.buildFragment()`.

### File Layout

```
[Column page 0 for column 0]
[Column page 1 for column 0]
...
[Column page 0 for column 1]
...
[Column metadata protobuf]
[Column metadata offset array]
[Footer — last 40 bytes]
```

### Footer (last 40 bytes)

```
Offset  Size     Field
──────  ───────  ────────────────────
-40     8        columnMetaStart (uint64 LE)
-32     8        columnMetaOffsetsStart (uint64 LE)
-24     8        globalBuffOffsetsStart (uint64 LE)
-16     4        numGlobalBuffers (uint32 LE)
-12     4        numColumns (uint32 LE)
-8      2        majorVersion (uint16 LE) — always 2
-6      2        minorVersion (uint16 LE) — 0 or 1
-4      4        Magic: "LANC" (ASCII)
```

### Column Metadata (protobuf per column)

```
field 1 (LEN)      name (UTF-8 string)
field 2 (LEN)      dtype string ("int64", "float64", "utf8", "bool", "fixed_size_list")
field 3 (VARINT)    nullable (0 or 1)
field 4 (FIXED64)   data_offset (byte offset within file)
field 5 (VARINT)    row_count
field 6 (VARINT)    data_size (bytes)
```

### Page Layout (Lance v2)

```
If nullable and nullCount > 0:
  [null bitmap: ceil(rowCount / 8) bytes]
  [alignment padding to dataOffsetInPage]
  [column data]
Else:
  [column data]
```

Column data format by type:
- **int8/16/32/64, uint8/16/32/64**: Packed native-endian values
- **float16/32/64**: Packed IEEE 754 values (float16 = 2 bytes)
- **utf8**: `[length-prefixed strings]` — each string preceded by 4-byte LE length
- **bool**: Bit-packed, 1 bit per value
- **fixed_size_list (vectors)**: `float32 × dimension × rowCount` (contiguous)

### Page Statistics

Extracted during footer parsing. Stored in `PageInfo`:
- `minValue`, `maxValue`: For numeric and string columns
- `nullCount`: Number of null values
- `rowCount`: Rows in this page
- These enable page-level skip without reading data.

---

## 3. Lance Dataset (multi-fragment directory)

```
{dataset}.lance/
├── _versions/
│   ├── _latest              # Current version number (plain text)
│   ├── 1.manifest           # Version 1 manifest (protobuf)
│   ├── 2.manifest           # Version 2 manifest
│   └── ...
└── data/
    ├── {uuid1}.lance         # Fragment 1 (single Lance file)
    ├── {uuid2}.lance         # Fragment 2
    └── ...
```

### `_latest` File

Plain text containing the current version number (e.g., "3"). Used for CAS via ETag.

### Manifest (protobuf)

```
field 1 (VARINT)    version number
field 2 (LEN)       repeated fragment entries
  field 1 (VARINT)    fragment id
  field 2 (LEN)       file_path (relative to data/)
  field 3 (VARINT)    physical_rows
field 3 (LEN)       schema (repeated SchemaField)
  field 1 (LEN)       name
  field 2 (LEN)       logical_type
  field 3 (VARINT)    field_id
  field 4 (VARINT)    parent_id
  field 5 (VARINT)    nullable
```

---

## 4. Partition Catalog

Stored in MasterDO durable storage. Enables O(1) fragment lookup by partition value.

### Structure (JSON, key: `partition-catalog:{table}`)

```json
{
  "column": "region",
  "entries": {
    "US": { "fragmentIds": [1, 3, 7], "exactPartition": true },
    "EU": { "fragmentIds": [2, 5], "exactPartition": true },
    "APAC": { "fragmentIds": [4, 6], "exactPartition": true }
  }
}
```

### `exactPartition` Flag

- `true` when all rows in these fragments have exactly this partition value (min === max). Used for eq/in filters.
- `false` when fragments contain a range of values. Falls back to min/max pruning.

### Supported Filter Ops for Catalog Pruning

| Op | Requires exactPartition | Behavior |
|----|------------------------|----------|
| `eq` | Yes | Return fragments for this value |
| `in` | Yes | Union fragments for all values in set |
| `neq` | Yes | All fragments EXCEPT this value |
| `not_in` | Yes | All fragments EXCEPT these values |
| Other | No | Falls through to fragment min/max pruning |

---

## 5. Vector Index (IVF-PQ)

Stored as `{r2Key}.ivf_pq.index` alongside the Lance file.

### Index Structure (binary)

Built and read by WASM (`ivf_pq.zig`). Contains:
- Partition centroids (float32 × nPartitions × dimension)
- PQ codebooks (float32 × nSubvectors × 256 × subDimension)
- Compressed codes (uint8 × nVectors × nSubvectors)
- Partition assignments (uint32 × nVectors)

### Search Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `topK` | required | 1-10000 | Number of results |
| `nprobe` | from config | 1-nPartitions | Partitions to search (more = better recall, slower) |
| `metric` | "cosine" | cosine/l2/dot | Distance metric |

---

## 6. Spill Format (R2)

External sort and large join intermediate data. Stored in `__spill/{uuid}/`.

### Spill Run Layout

```
[4-byte frame length (uint32 LE)]
[QMCB binary frame]
[4-byte frame length]
[QMCB binary frame]
...
```

Each frame is a sorted QMCB batch. Runs are merged via k-way merge during read-back.

### Lifecycle

- Created during query execution (ExternalSortOperator, HashJoinOperator)
- Cleaned up with `bucket.delete()` after query completes (with 10s timeout)
- Location: `__spill/{randomUUID}/run_{i}`

---

## 7. Streaming Wire Format (RPC)

Used by `streamRpc()` for streaming results from QueryDO to client.

```
[4-byte frame length (uint32 LE)][QMCB binary]
[4-byte frame length (uint32 LE)][QMCB binary]
...
[stream ends]
```

Client reads: 4 bytes → frame length → read exactly that many bytes → decodeColumnarRun() → Row[].

---

## 8. Type System

### DataType (string enum)

```
int8, int16, int32, int64
uint8, uint16, uint32, uint64
float16, float32, float64
utf8, binary, bool, fixed_size_list
```

### Type Promotion (WASM registration)

| Source | Target | Why |
|--------|--------|-----|
| int8, int16, int32 | int64 | WASM SQL engine only handles int64 |
| float16, float32 | float64 | WASM SQL engine only handles float64 |
| All others | unchanged | |

### Row Value Types (JS)

| DataType | JS Type | Notes |
|----------|---------|-------|
| int8-int64, uint8-uint64 | `bigint` | Except int32/uint32 which may be `number` |
| float16-float64 | `number` | |
| utf8 | `string` | |
| bool | `boolean` | |
| fixed_size_list | `Float32Array` | Vector embeddings |
| null | `null` | |
