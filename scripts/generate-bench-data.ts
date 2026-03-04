#!/usr/bin/env npx tsx
/**
 * Generate benchmark data files (Parquet + Iceberg) for QueryMode benchmarking.
 *
 * Creates multi-row-group Parquet files with column statistics so that
 * page-level skipping via footer min/max works in benchmarks.
 *
 * Usage: npx tsx scripts/generate-bench-data.ts
 */

import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";

const OUT_DIR = join(import.meta.dirname, "../wasm/tests/fixtures/generated");
const ROW_GROUP_SIZE = 10_000; // 10K rows per row group → page-level skipping

// --- Thrift compact protocol encoder ---

function encodeVarintArr(value: number): number[] {
  const bytes: number[] = [];
  let v = value >>> 0;
  while (v > 0x7f) { bytes.push((v & 0x7f) | 0x80); v >>>= 7; }
  bytes.push(v);
  return bytes;
}

function encodeZigzagVarint(value: number): number[] {
  const zigzag = (value << 1) ^ (value >> 31);
  return encodeVarintArr(zigzag >>> 0);
}

function encodeZigzag64(value: bigint): number[] {
  const zigzag = (value << 1n) ^ (value >> 63n);
  return encodeBigVarint(zigzag < 0n ? -zigzag : zigzag);
}

function encodeBigVarint(value: bigint): number[] {
  const bytes: number[] = [];
  let v = value;
  while (v > 0x7fn) { bytes.push(Number(v & 0x7fn) | 0x80); v >>= 7n; }
  bytes.push(Number(v));
  return bytes;
}

function thriftField(fieldId: number, lastFieldId: number, typeId: number): { bytes: number[]; newLastId: number } {
  const delta = fieldId - lastFieldId;
  if (delta > 0 && delta < 16) {
    return { bytes: [(delta << 4) | typeId], newLastId: fieldId };
  }
  const zigzag = (fieldId << 1) ^ (fieldId >> 31);
  const varint = encodeVarintArr(zigzag);
  return { bytes: [typeId, ...varint], newLastId: fieldId };
}

function thriftListHeader(size: number, elemType: number): number[] {
  if (size < 15) return [(size << 4) | elemType];
  return [0xf0 | elemType, ...encodeVarintArr(size)];
}

function thriftBinary(data: Buffer): number[] {
  return [...encodeVarintArr(data.length), ...data];
}

// --- Column value encoding ---

function encodeColumnValues(type: string, values: (number | bigint | string)[]): Buffer {
  if (type === "int64") {
    const buf = Buffer.alloc(values.length * 8);
    for (let i = 0; i < values.length; i++) buf.writeBigInt64LE(BigInt(values[i] as number), i * 8);
    return buf;
  }
  if (type === "float64") {
    const buf = Buffer.alloc(values.length * 8);
    for (let i = 0; i < values.length; i++) buf.writeDoubleLE(values[i] as number, i * 8);
    return buf;
  }
  if (type === "utf8") {
    const parts: Buffer[] = [];
    for (const v of values) {
      const str = Buffer.from(v as string, "utf8");
      const lenBuf = Buffer.alloc(4);
      lenBuf.writeUInt32LE(str.length, 0);
      parts.push(lenBuf, str);
    }
    return Buffer.concat(parts);
  }
  throw new Error(`Unsupported type: ${type}`);
}

function encodeStatValue(type: string, value: number | bigint | string): Buffer {
  if (type === "int64") { const b = Buffer.alloc(8); b.writeBigInt64LE(BigInt(value as number), 0); return b; }
  if (type === "float64") { const b = Buffer.alloc(8); b.writeDoubleLE(value as number, 0); return b; }
  return Buffer.from(String(value), "utf8");
}

function physicalType(type: string): number {
  return type === "int64" ? 2 : type === "float64" ? 5 : 6;
}

// --- Data page header ---

function encodeDataPageHeader(uncompressedSize: number, numValues: number): Buffer {
  const bytes: number[] = [];
  let lastId = 0;
  let f = thriftField(1, lastId, 5); lastId = f.newLastId;
  bytes.push(...f.bytes, ...encodeZigzagVarint(0)); // type = DATA_PAGE
  f = thriftField(2, lastId, 5); lastId = f.newLastId;
  bytes.push(...f.bytes, ...encodeZigzagVarint(uncompressedSize));
  f = thriftField(3, lastId, 5); lastId = f.newLastId;
  bytes.push(...f.bytes, ...encodeZigzagVarint(uncompressedSize));
  f = thriftField(5, lastId, 12); lastId = f.newLastId;
  bytes.push(...f.bytes);
  {
    let il = 0;
    let fi = thriftField(1, il, 5); il = fi.newLastId;
    bytes.push(...fi.bytes, ...encodeZigzagVarint(numValues));
    fi = thriftField(2, il, 5); il = fi.newLastId;
    bytes.push(...fi.bytes, ...encodeZigzagVarint(0)); // PLAIN
    fi = thriftField(3, il, 5); il = fi.newLastId;
    bytes.push(...fi.bytes, ...encodeZigzagVarint(0));
    fi = thriftField(4, il, 5); il = fi.newLastId;
    bytes.push(...fi.bytes, ...encodeZigzagVarint(0));
    bytes.push(0x00);
  }
  bytes.push(0x00);
  return Buffer.from(bytes);
}

// --- Statistics encoding (Thrift struct inside ColumnMetaData) ---

function encodeStatistics(minVal: Buffer, maxVal: Buffer, numValues: number): number[] {
  const bytes: number[] = [];
  let sl = 0;
  // field 3 (i64): num_values (legacy but some readers want it)
  // field 5 (binary): max_value
  let sf = thriftField(5, sl, 8); sl = sf.newLastId;
  bytes.push(...sf.bytes, ...thriftBinary(maxVal));
  // field 6 (binary): min_value
  sf = thriftField(6, sl, 8); sl = sf.newLastId;
  bytes.push(...sf.bytes, ...thriftBinary(minVal));
  bytes.push(0x00);
  return bytes;
}

// --- Multi-row-group Parquet writer ---

interface Column {
  name: string;
  type: "int64" | "float64" | "utf8";
  values: (number | bigint | string)[];
}

function writeParquet(columns: Column[], rowGroupSize = ROW_GROUP_SIZE): Buffer {
  const numRows = columns[0].values.length;
  const numRowGroups = Math.ceil(numRows / rowGroupSize);
  const parts: Buffer[] = [];
  let fileOffset = 0;

  const addPart = (buf: Buffer) => { parts.push(buf); fileOffset += buf.length; };
  addPart(Buffer.from("PAR1"));

  // Track per-row-group column chunk metadata
  const rowGroupMetas: {
    numRows: number;
    chunks: {
      type: number; codec: number; numValues: number;
      dataPageOffset: number; totalCompressed: number; totalUncompressed: number;
      pathInSchema: string[];
      minValue: Buffer; maxValue: Buffer;
    }[];
  }[] = [];

  for (let rg = 0; rg < numRowGroups; rg++) {
    const startRow = rg * rowGroupSize;
    const endRow = Math.min(startRow + rowGroupSize, numRows);
    const rgRows = endRow - startRow;
    const chunks: typeof rowGroupMetas[0]["chunks"] = [];

    for (const col of columns) {
      const slice = col.values.slice(startRow, endRow);
      const dataPageOffset = fileOffset;
      const valBuf = encodeColumnValues(col.type, slice);
      const pageHeader = encodeDataPageHeader(valBuf.length, rgRows);
      addPart(pageHeader);
      addPart(valBuf);
      const totalSize = pageHeader.length + valBuf.length;

      // Compute min/max for this slice
      let minV = slice[0], maxV = slice[0];
      for (let i = 1; i < slice.length; i++) {
        if (slice[i] < minV) minV = slice[i];
        if (slice[i] > maxV) maxV = slice[i];
      }

      chunks.push({
        type: physicalType(col.type),
        codec: 0,
        numValues: rgRows,
        dataPageOffset,
        totalCompressed: totalSize,
        totalUncompressed: totalSize,
        pathInSchema: [col.name],
        minValue: encodeStatValue(col.type, minV),
        maxValue: encodeStatValue(col.type, maxV),
      });
    }
    rowGroupMetas.push({ numRows: rgRows, chunks });
  }

  // Build Thrift footer
  const footer = encodeParquetFooter(columns, rowGroupMetas, numRows);
  addPart(footer);
  const footerLenBuf = Buffer.alloc(4);
  footerLenBuf.writeUInt32LE(footer.length, 0);
  addPart(footerLenBuf);
  addPart(Buffer.from("PAR1"));

  return Buffer.concat(parts);
}

function encodeParquetFooter(
  columns: Column[],
  rowGroups: {
    numRows: number;
    chunks: {
      type: number; codec: number; numValues: number;
      dataPageOffset: number; totalCompressed: number; totalUncompressed: number;
      pathInSchema: string[];
      minValue: Buffer; maxValue: Buffer;
    }[];
  }[],
  totalRows: number,
): Buffer {
  const bytes: number[] = [];
  let lastId = 0;

  // field 1 (i32): version = 2
  let f = thriftField(1, lastId, 5); lastId = f.newLastId;
  bytes.push(...f.bytes, ...encodeZigzagVarint(2));

  // field 2 (list<SchemaElement>): schema
  f = thriftField(2, lastId, 9); lastId = f.newLastId;
  bytes.push(...f.bytes, ...thriftListHeader(columns.length + 1, 12));

  // Root schema element
  {
    let sl = 0;
    let sf = thriftField(4, sl, 8); sl = sf.newLastId;
    const nameBytes = Buffer.from("schema");
    bytes.push(...sf.bytes, ...thriftBinary(nameBytes));
    sf = thriftField(5, sl, 5); sl = sf.newLastId;
    bytes.push(...sf.bytes, ...encodeZigzagVarint(columns.length));
    bytes.push(0x00);
  }
  // Leaf schema elements
  for (const col of columns) {
    let sl = 0;
    let sf = thriftField(1, sl, 5); sl = sf.newLastId;
    bytes.push(...sf.bytes, ...encodeZigzagVarint(physicalType(col.type)));
    sf = thriftField(3, sl, 5); sl = sf.newLastId;
    bytes.push(...sf.bytes, ...encodeZigzagVarint(0)); // REQUIRED
    sf = thriftField(4, sl, 8); sl = sf.newLastId;
    bytes.push(...sf.bytes, ...thriftBinary(Buffer.from(col.name)));
    if (col.type === "utf8") {
      sf = thriftField(6, sl, 5); sl = sf.newLastId;
      bytes.push(...sf.bytes, ...encodeZigzagVarint(0)); // UTF8
    }
    bytes.push(0x00);
  }

  // field 3 (i64): num_rows
  f = thriftField(3, lastId, 6); lastId = f.newLastId;
  bytes.push(...f.bytes, ...encodeZigzag64(BigInt(totalRows)));

  // field 4 (list<RowGroup>): row_groups
  f = thriftField(4, lastId, 9); lastId = f.newLastId;
  bytes.push(...f.bytes, ...thriftListHeader(rowGroups.length, 12));

  for (const rg of rowGroups) {
    let rgl = 0;
    // field 1 (list<ColumnChunk>)
    let rgf = thriftField(1, rgl, 9); rgl = rgf.newLastId;
    bytes.push(...rgf.bytes, ...thriftListHeader(rg.chunks.length, 12));

    for (const cm of rg.chunks) {
      let ccl = 0;
      // field 2 (i64): file_offset
      let ccf = thriftField(2, ccl, 6); ccl = ccf.newLastId;
      bytes.push(...ccf.bytes, ...encodeZigzag64(BigInt(cm.dataPageOffset)));
      // field 3 (struct): column_metadata
      ccf = thriftField(3, ccl, 12); ccl = ccf.newLastId;
      bytes.push(...ccf.bytes);
      {
        let cml = 0;
        let cmf = thriftField(1, cml, 5); cml = cmf.newLastId;
        bytes.push(...cmf.bytes, ...encodeZigzagVarint(cm.type));
        cmf = thriftField(2, cml, 9); cml = cmf.newLastId;
        bytes.push(...cmf.bytes, (1 << 4) | 5, ...encodeZigzagVarint(0)); // PLAIN
        cmf = thriftField(3, cml, 9); cml = cmf.newLastId;
        bytes.push(...cmf.bytes, (cm.pathInSchema.length << 4) | 8);
        for (const p of cm.pathInSchema) bytes.push(...thriftBinary(Buffer.from(p)));
        cmf = thriftField(4, cml, 5); cml = cmf.newLastId;
        bytes.push(...cmf.bytes, ...encodeZigzagVarint(0)); // UNCOMPRESSED
        cmf = thriftField(5, cml, 6); cml = cmf.newLastId;
        bytes.push(...cmf.bytes, ...encodeZigzag64(BigInt(cm.numValues)));
        cmf = thriftField(6, cml, 6); cml = cmf.newLastId;
        bytes.push(...cmf.bytes, ...encodeZigzag64(BigInt(cm.totalUncompressed)));
        cmf = thriftField(7, cml, 6); cml = cmf.newLastId;
        bytes.push(...cmf.bytes, ...encodeZigzag64(BigInt(cm.totalCompressed)));
        cmf = thriftField(9, cml, 6); cml = cmf.newLastId;
        bytes.push(...cmf.bytes, ...encodeZigzag64(BigInt(cm.dataPageOffset)));
        // field 12 (struct): statistics — min/max values
        cmf = thriftField(12, cml, 12); cml = cmf.newLastId;
        bytes.push(...cmf.bytes, ...encodeStatistics(cm.minValue, cm.maxValue, cm.numValues));
        bytes.push(0x00); // end ColumnMetaData
      }
      bytes.push(0x00); // end ColumnChunk
    }

    // field 2 (i64): total_byte_size
    const totalBytes = rg.chunks.reduce((s, c) => s + c.totalCompressed, 0);
    rgf = thriftField(2, rgl, 6); rgl = rgf.newLastId;
    bytes.push(...rgf.bytes, ...encodeZigzag64(BigInt(totalBytes)));
    // field 3 (i64): num_rows
    rgf = thriftField(3, rgl, 6); rgl = rgf.newLastId;
    bytes.push(...rgf.bytes, ...encodeZigzag64(BigInt(rg.numRows)));
    bytes.push(0x00); // end RowGroup
  }

  bytes.push(0x00); // end FileMetaData
  return Buffer.from(bytes);
}

// --- Iceberg metadata generator ---

function generateIcebergMetadata(
  tableName: string,
  schema: { name: string; type: string; icebergType: string }[],
  parquetPath: string,
  numRows: number,
): { metadataJson: string; manifestListAvro: Buffer } {
  const snapshotId = Date.now();
  const manifestListPath = `metadata/snap-${snapshotId}-0.avro`;

  const metadata = {
    "format-version": 2,
    "table-uuid": crypto.randomUUID(),
    "location": `s3://querymode-data/${tableName}`,
    "current-schema-id": 0,
    "current-snapshot-id": snapshotId,
    "schemas": [{
      "schema-id": 0,
      "type": "struct",
      "fields": schema.map((f, i) => ({
        id: i + 1, name: f.name, required: true, type: f.icebergType,
      })),
    }],
    "snapshots": [{
      "snapshot-id": snapshotId,
      "timestamp-ms": Date.now(),
      "manifest-list": manifestListPath,
      "summary": { operation: "append", "added-data-files": "1", "total-records": String(numRows) },
    }],
    "partition-specs": [{ "spec-id": 0, "fields": [] }],
    "default-spec-id": 0,
    "last-partition-id": 0,
    "properties": {},
    "sort-orders": [{ "order-id": 0, "fields": [] }],
    "default-sort-order-id": 0,
  };

  const pathStr = `data/${parquetPath}`;
  // Avro zigzag-encoded long for string length: (n << 1) for small positive n
  const pathBytes = Buffer.from(pathStr, "utf8");
  const zigzagLen = encodeVarintArr(pathBytes.length << 1);
  const schemaJson = '{"type":"record","name":"manifest_file","fields":[{"name":"manifest_path","type":"string"}]}';
  const schemaBytes = Buffer.from(schemaJson, "utf8");
  const zigzagSchemaLen = encodeVarintArr(schemaBytes.length << 1);
  const syncMarker = Buffer.alloc(16); // 16-byte sync marker (zeros)
  const avroContent = Buffer.concat([
    // Avro container header
    Buffer.from("Obj\x01"),                   // magic + version
    Buffer.from([0x02]),                      // map: 1 entry (zigzag 1 = 2)
    // map entry: "avro.schema" -> schema JSON
    Buffer.from([0x16]),                      // key length zigzag(11) = 22
    Buffer.from("avro.schema"),
    Buffer.from(zigzagSchemaLen),              // schema value length (zigzag)
    schemaBytes,
    Buffer.from([0x00]),                      // end of map
    syncMarker,
    // Single data block: 1 object (block count = zigzag(1) = 2)
    Buffer.from([0x02]),                      // block count: 1
    Buffer.from(encodeVarintArr((pathBytes.length + zigzagLen.length) << 1)), // block byte size (zigzag)
    Buffer.from(zigzagLen),                   // string length (zigzag)
    pathBytes,                                // string data
    syncMarker,
  ]);

  return { metadataJson: JSON.stringify(metadata, null, 2), manifestListAvro: avroContent };
}

// --- Main ---

async function main(): Promise<void> {
  if (!existsSync(OUT_DIR)) mkdirSync(OUT_DIR, { recursive: true });

  const ROWS_100K = 100_000;
  const ROWS_1M = 1_000_000;

  // 100K rows, 3 columns, 10 row groups of 10K each
  console.log("Generating bench_100k_3col.parquet (10 row groups × 10K rows)...");
  {
    const ids: bigint[] = [], values: number[] = [], categories: string[] = [];
    const cats = ["alpha", "beta", "gamma", "delta", "epsilon"];
    for (let i = 0; i < ROWS_100K; i++) {
      ids.push(BigInt(i));
      values.push(Math.random() * 1000);
      categories.push(cats[i % cats.length]);
    }
    const buf = writeParquet([
      { name: "id", type: "int64", values: ids },
      { name: "value", type: "float64", values },
      { name: "category", type: "utf8", values: categories },
    ]);
    writeFileSync(join(OUT_DIR, "bench_100k_3col.parquet"), buf);
    console.log(`  Written: ${(buf.length / 1024 / 1024).toFixed(1)} MB`);
  }

  // 100K rows, 2 numeric columns, 10 row groups
  console.log("Generating bench_100k_numeric.parquet (10 row groups × 10K rows)...");
  {
    const ids: bigint[] = [], values: number[] = [];
    for (let i = 0; i < ROWS_100K; i++) {
      ids.push(BigInt(i));
      values.push(Math.random() * 10000);
    }
    const buf = writeParquet([
      { name: "id", type: "int64", values: ids },
      { name: "value", type: "float64", values },
    ]);
    writeFileSync(join(OUT_DIR, "bench_100k_numeric.parquet"), buf);
    console.log(`  Written: ${(buf.length / 1024 / 1024).toFixed(1)} MB`);
  }

  // 1M rows, 2 numeric columns, 100 row groups of 10K each
  console.log("Generating bench_1m_numeric.parquet (100 row groups × 10K rows)...");
  {
    const ids: bigint[] = [], values: number[] = [];
    for (let i = 0; i < ROWS_1M; i++) {
      ids.push(BigInt(i));
      values.push(Math.random() * 100000);
    }
    const buf = writeParquet([
      { name: "id", type: "int64", values: ids },
      { name: "value", type: "float64", values },
    ]);
    writeFileSync(join(OUT_DIR, "bench_1m_numeric.parquet"), buf);
    console.log(`  Written: ${(buf.length / 1024 / 1024).toFixed(1)} MB`);
  }

  // Iceberg table (100K rows, 10 row groups)
  console.log("Generating bench_iceberg_100k/ (10 row groups × 10K rows)...");
  {
    const iceDir = join(OUT_DIR, "bench_iceberg_100k");
    mkdirSync(join(iceDir, "metadata"), { recursive: true });
    mkdirSync(join(iceDir, "data"), { recursive: true });

    const ids: bigint[] = [], values: number[] = [];
    for (let i = 0; i < ROWS_100K; i++) {
      ids.push(BigInt(i));
      values.push(Math.random() * 1000);
    }
    const parquetBuf = writeParquet([
      { name: "id", type: "int64", values: ids },
      { name: "value", type: "float64", values },
    ]);
    writeFileSync(join(iceDir, "data", "00000-0.parquet"), parquetBuf);

    const { metadataJson, manifestListAvro } = generateIcebergMetadata(
      "bench_iceberg_100k",
      [
        { name: "id", type: "int64", icebergType: "long" },
        { name: "value", type: "float64", icebergType: "double" },
      ],
      "00000-0.parquet",
      ROWS_100K,
    );
    writeFileSync(join(iceDir, "metadata", "v1.metadata.json"), metadataJson);
    writeFileSync(join(iceDir, "metadata", `snap-${Date.now()}-0.avro`), manifestListAvro);
    console.log(`  Parquet: ${(parquetBuf.length / 1024 / 1024).toFixed(1)} MB`);
  }

  console.log("\nAll benchmark data generated in", OUT_DIR);
}

main().catch(console.error);
