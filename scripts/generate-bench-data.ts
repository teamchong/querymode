#!/usr/bin/env npx tsx
/**
 * Generate benchmark data files (Parquet + Iceberg) for QueryMode benchmarking.
 *
 * Creates synthetic Parquet files using Apache Arrow IPC format written by hand,
 * then seeds them to local R2 via wrangler CLI.
 *
 * Usage: npx tsx scripts/generate-bench-data.ts
 */

import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";

const OUT_DIR = join(import.meta.dirname, "../wasm/tests/fixtures/generated");

// --- Parquet writer (minimal, PLAIN encoding, no compression) ---

function writeParquet(
  columns: { name: string; type: "int64" | "float64" | "utf8"; values: (number | bigint | string)[] }[],
): Buffer {
  const numRows = columns[0].values.length;
  const parts: Buffer[] = [];

  // PAR1 magic
  parts.push(Buffer.from("PAR1"));

  // Write column chunks
  const columnChunkMetas: {
    type: number; codec: number; numValues: number;
    dataPageOffset: number; totalCompressed: number; totalUncompressed: number;
    pathInSchema: string[];
  }[] = [];

  for (const col of columns) {
    const dataPageOffset = parts.reduce((s, b) => s + b.length, 0);

    // Build data page: page header (Thrift) + values
    const valBuf = encodeColumnValues(col.type, col.values);

    // Data page header (Thrift compact protocol)
    const pageHeader = encodeDataPageHeader(valBuf.length, numRows);
    parts.push(pageHeader);
    parts.push(valBuf);

    const totalSize = pageHeader.length + valBuf.length;
    columnChunkMetas.push({
      type: col.type === "int64" ? 2 : col.type === "float64" ? 5 : 6,
      codec: 0, // UNCOMPRESSED
      numValues: numRows,
      dataPageOffset,
      totalCompressed: totalSize,
      totalUncompressed: totalSize,
      pathInSchema: [col.name],
    });
  }

  // Build Thrift footer
  const footer = encodeParquetFooter(columns, columnChunkMetas, numRows);
  parts.push(footer);

  // Footer length + PAR1
  const footerLenBuf = Buffer.alloc(4);
  footerLenBuf.writeUInt32LE(footer.length, 0);
  parts.push(footerLenBuf);
  parts.push(Buffer.from("PAR1"));

  return Buffer.concat(parts);
}

function encodeColumnValues(type: string, values: (number | bigint | string)[]): Buffer {
  if (type === "int64") {
    const buf = Buffer.alloc(values.length * 8);
    for (let i = 0; i < values.length; i++) {
      buf.writeBigInt64LE(BigInt(values[i] as number), i * 8);
    }
    return buf;
  }
  if (type === "float64") {
    const buf = Buffer.alloc(values.length * 8);
    for (let i = 0; i < values.length; i++) {
      buf.writeDoubleLE(values[i] as number, i * 8);
    }
    return buf;
  }
  if (type === "utf8") {
    // BYTE_ARRAY encoding: each value is 4-byte length + bytes
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

// --- Thrift compact protocol encoder ---

function thriftField(fieldId: number, lastFieldId: number, typeId: number): { bytes: number[]; newLastId: number } {
  const delta = fieldId - lastFieldId;
  if (delta > 0 && delta < 16) {
    return { bytes: [(delta << 4) | typeId], newLastId: fieldId };
  }
  // Full field ID
  const zigzag = (fieldId << 1) ^ (fieldId >> 31);
  const varint = encodeVarintArr(zigzag);
  return { bytes: [typeId, ...varint], newLastId: fieldId };
}

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

function encodeDataPageHeader(uncompressedSize: number, numValues: number): Buffer {
  // PageHeader Thrift struct:
  //   field 1 (i32): type = 0 (DATA_PAGE)
  //   field 2 (i32): uncompressed_page_size
  //   field 3 (i32): compressed_page_size
  //   field 5 (struct): DataPageHeader
  //     field 1 (i32): num_values
  //     field 2 (i32): encoding = 0 (PLAIN)
  //     field 3 (i32): definition_level_encoding = 0 (PLAIN)
  //     field 4 (i32): repetition_level_encoding = 0 (PLAIN)
  const bytes: number[] = [];
  let lastId = 0;

  // field 1: type = DATA_PAGE (0)
  let f = thriftField(1, lastId, 5); lastId = f.newLastId;
  bytes.push(...f.bytes, ...encodeZigzagVarint(0));

  // field 2: uncompressed_page_size
  f = thriftField(2, lastId, 5); lastId = f.newLastId;
  bytes.push(...f.bytes, ...encodeZigzagVarint(uncompressedSize));

  // field 3: compressed_page_size
  f = thriftField(3, lastId, 5); lastId = f.newLastId;
  bytes.push(...f.bytes, ...encodeZigzagVarint(uncompressedSize));

  // field 5: DataPageHeader struct
  f = thriftField(5, lastId, 12); lastId = f.newLastId;
  bytes.push(...f.bytes);
  {
    let innerLast = 0;
    // num_values
    let fi = thriftField(1, innerLast, 5); innerLast = fi.newLastId;
    bytes.push(...fi.bytes, ...encodeZigzagVarint(numValues));
    // encoding = PLAIN (0)
    fi = thriftField(2, innerLast, 5); innerLast = fi.newLastId;
    bytes.push(...fi.bytes, ...encodeZigzagVarint(0));
    // def level encoding = PLAIN (0)
    fi = thriftField(3, innerLast, 5); innerLast = fi.newLastId;
    bytes.push(...fi.bytes, ...encodeZigzagVarint(0));
    // rep level encoding = PLAIN (0)
    fi = thriftField(4, innerLast, 5); innerLast = fi.newLastId;
    bytes.push(...fi.bytes, ...encodeZigzagVarint(0));
    bytes.push(0x00); // struct end
  }

  bytes.push(0x00); // struct end
  return Buffer.from(bytes);
}

function encodeParquetFooter(
  columns: { name: string; type: string }[],
  chunkMetas: {
    type: number; codec: number; numValues: number;
    dataPageOffset: number; totalCompressed: number; totalUncompressed: number;
    pathInSchema: string[];
  }[],
  numRows: number,
): Buffer {
  const bytes: number[] = [];
  let lastId = 0;

  // field 1 (i32): version = 2
  let f = thriftField(1, lastId, 5); lastId = f.newLastId;
  bytes.push(...f.bytes, ...encodeZigzagVarint(2));

  // field 2 (list<SchemaElement>): schema
  f = thriftField(2, lastId, 9); lastId = f.newLastId;
  bytes.push(...f.bytes);
  // list header: size = columns.length + 1 (root + leaves), elemType = 12 (struct)
  const schemaSize = columns.length + 1;
  if (schemaSize < 15) {
    bytes.push((schemaSize << 4) | 12);
  } else {
    bytes.push(0xfc, ...encodeVarintArr(schemaSize));
  }
  // Root schema element
  {
    let sl = 0;
    // field 4 (binary): name = "schema"
    let sf = thriftField(4, sl, 8); sl = sf.newLastId;
    const nameBytes = Buffer.from("schema");
    bytes.push(...sf.bytes, ...encodeVarintArr(nameBytes.length), ...nameBytes);
    // field 5 (i32): num_children
    sf = thriftField(5, sl, 5); sl = sf.newLastId;
    bytes.push(...sf.bytes, ...encodeZigzagVarint(columns.length));
    bytes.push(0x00);
  }
  // Leaf schema elements
  for (const col of columns) {
    let sl = 0;
    // field 1 (i32): type
    const physType = col.type === "int64" ? 2 : col.type === "float64" ? 5 : 6;
    let sf = thriftField(1, sl, 5); sl = sf.newLastId;
    bytes.push(...sf.bytes, ...encodeZigzagVarint(physType));
    // field 3 (i32): repetition_type = REQUIRED (0)
    sf = thriftField(3, sl, 5); sl = sf.newLastId;
    bytes.push(...sf.bytes, ...encodeZigzagVarint(0));
    // field 4 (binary): name
    sf = thriftField(4, sl, 8); sl = sf.newLastId;
    const nameB = Buffer.from(col.name);
    bytes.push(...sf.bytes, ...encodeVarintArr(nameB.length), ...nameB);
    // field 6 (i32): converted_type (0 = UTF8 for BYTE_ARRAY)
    if (col.type === "utf8") {
      sf = thriftField(6, sl, 5); sl = sf.newLastId;
      bytes.push(...sf.bytes, ...encodeZigzagVarint(0)); // UTF8
    }
    bytes.push(0x00);
  }

  // field 3 (i64): num_rows
  f = thriftField(3, lastId, 6); lastId = f.newLastId;
  bytes.push(...f.bytes, ...encodeZigzag64(BigInt(numRows)));

  // field 4 (list<RowGroup>): row_groups — 1 row group
  f = thriftField(4, lastId, 9); lastId = f.newLastId;
  bytes.push(...f.bytes);
  bytes.push((1 << 4) | 12); // list of 1 struct

  // RowGroup
  {
    let rgl = 0;
    // field 1 (list<ColumnChunk>): columns
    let rgf = thriftField(1, rgl, 9); rgl = rgf.newLastId;
    bytes.push(...rgf.bytes);
    if (chunkMetas.length < 15) {
      bytes.push((chunkMetas.length << 4) | 12);
    } else {
      bytes.push(0xfc, ...encodeVarintArr(chunkMetas.length));
    }

    for (const cm of chunkMetas) {
      // ColumnChunk struct
      let ccl = 0;
      // field 2 (i64): file_offset
      let ccf = thriftField(2, ccl, 6); ccl = ccf.newLastId;
      bytes.push(...ccf.bytes, ...encodeZigzag64(BigInt(cm.dataPageOffset)));
      // field 3 (struct): column_metadata
      ccf = thriftField(3, ccl, 12); ccl = ccf.newLastId;
      bytes.push(...ccf.bytes);
      {
        let cml = 0;
        // field 1 (i32): type
        let cmf = thriftField(1, cml, 5); cml = cmf.newLastId;
        bytes.push(...cmf.bytes, ...encodeZigzagVarint(cm.type));
        // field 2 (list<i32>): encodings = [PLAIN]
        cmf = thriftField(2, cml, 9); cml = cmf.newLastId;
        bytes.push(...cmf.bytes, (1 << 4) | 5, ...encodeZigzagVarint(0));
        // field 3 (list<string>): path_in_schema
        cmf = thriftField(3, cml, 9); cml = cmf.newLastId;
        bytes.push(...cmf.bytes, (cm.pathInSchema.length << 4) | 8);
        for (const p of cm.pathInSchema) {
          const pb = Buffer.from(p);
          bytes.push(...encodeVarintArr(pb.length), ...pb);
        }
        // field 4 (i32): codec = UNCOMPRESSED
        cmf = thriftField(4, cml, 5); cml = cmf.newLastId;
        bytes.push(...cmf.bytes, ...encodeZigzagVarint(0));
        // field 5 (i64): num_values
        cmf = thriftField(5, cml, 6); cml = cmf.newLastId;
        bytes.push(...cmf.bytes, ...encodeZigzag64(BigInt(cm.numValues)));
        // field 6 (i64): total_uncompressed_size
        cmf = thriftField(6, cml, 6); cml = cmf.newLastId;
        bytes.push(...cmf.bytes, ...encodeZigzag64(BigInt(cm.totalUncompressed)));
        // field 7 (i64): total_compressed_size
        cmf = thriftField(7, cml, 6); cml = cmf.newLastId;
        bytes.push(...cmf.bytes, ...encodeZigzag64(BigInt(cm.totalCompressed)));
        // field 9 (i64): data_page_offset
        cmf = thriftField(9, cml, 6); cml = cmf.newLastId;
        bytes.push(...cmf.bytes, ...encodeZigzag64(BigInt(cm.dataPageOffset)));
        bytes.push(0x00); // end ColumnMetaData
      }
      bytes.push(0x00); // end ColumnChunk
    }

    // field 2 (i64): total_byte_size (approximate)
    const totalBytes = chunkMetas.reduce((s, c) => s + c.totalCompressed, 0);
    rgf = thriftField(2, rgl, 6); rgl = rgf.newLastId;
    bytes.push(...rgf.bytes, ...encodeZigzag64(BigInt(totalBytes)));

    // field 3 (i64): num_rows
    rgf = thriftField(3, rgl, 6); rgl = rgf.newLastId;
    bytes.push(...rgf.bytes, ...encodeZigzag64(BigInt(numRows)));

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

  // Minimal Avro manifest list — embed the parquet path as a recognizable string
  // The extractParquetPathsFromManifest function uses regex to find .parquet paths
  const pathStr = `data/${parquetPath}`;
  const avroContent = Buffer.from(
    `Obj\x01\x04\x16avro.schema\xb2\x01{"type":"record","name":"manifest_file","fields":[{"name":"manifest_path","type":"string"}]}\x00` +
    `\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00` +
    `\x02${String.fromCharCode(pathStr.length * 2)}${pathStr}` +
    `\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`
  );

  return { metadataJson: JSON.stringify(metadata, null, 2), manifestListAvro: avroContent };
}

// --- Main ---

async function main(): Promise<void> {
  if (!existsSync(OUT_DIR)) mkdirSync(OUT_DIR, { recursive: true });

  const ROWS_100K = 100_000;
  const ROWS_1M = 1_000_000;

  // --- Generate 100K row Parquet (3 columns: id int64, value float64, category utf8) ---
  console.log("Generating bench_100k_3col.parquet ...");
  {
    const ids: bigint[] = [];
    const values: number[] = [];
    const categories: string[] = [];
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

  // --- Generate 100K row Parquet (2 numeric columns only — scan speed test) ---
  console.log("Generating bench_100k_numeric.parquet ...");
  {
    const ids: bigint[] = [];
    const values: number[] = [];
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

  // --- Generate 1M row Parquet (2 numeric columns — stress test) ---
  console.log("Generating bench_1m_numeric.parquet ...");
  {
    const ids: bigint[] = [];
    const values: number[] = [];
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

  // --- Generate Iceberg table (100K rows) ---
  console.log("Generating bench_iceberg_100k/ ...");
  {
    const iceDir = join(OUT_DIR, "bench_iceberg_100k");
    mkdirSync(join(iceDir, "metadata"), { recursive: true });
    mkdirSync(join(iceDir, "data"), { recursive: true });

    const ids: bigint[] = [];
    const values: number[] = [];
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
    writeFileSync(
      join(iceDir, "metadata", `snap-${Date.now()}-0.avro`),
      manifestListAvro,
    );

    console.log(`  Parquet: ${(parquetBuf.length / 1024 / 1024).toFixed(1)} MB`);
  }

  console.log("\nAll benchmark data generated in", OUT_DIR);
}

main().catch(console.error);
