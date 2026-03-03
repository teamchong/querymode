import { describe, it, expect } from "vitest";
import { parseIcebergMetadata, extractParquetPathsFromManifest, icebergTypeToDataType } from "./iceberg.js";

describe("icebergTypeToDataType", () => {
  it("maps basic types correctly", () => {
    expect(icebergTypeToDataType("boolean")).toBe("bool");
    expect(icebergTypeToDataType("int")).toBe("int32");
    expect(icebergTypeToDataType("long")).toBe("int64");
    expect(icebergTypeToDataType("float")).toBe("float32");
    expect(icebergTypeToDataType("double")).toBe("float64");
    expect(icebergTypeToDataType("string")).toBe("utf8");
    expect(icebergTypeToDataType("binary")).toBe("binary");
    expect(icebergTypeToDataType("timestamp")).toBe("int64");
    expect(icebergTypeToDataType("timestamptz")).toBe("int64");
    expect(icebergTypeToDataType("date")).toBe("int32");
    expect(icebergTypeToDataType("uuid")).toBe("utf8");
    expect(icebergTypeToDataType("decimal")).toBe("float64");
    expect(icebergTypeToDataType("time")).toBe("int64");
    expect(icebergTypeToDataType("fixed")).toBe("binary");
  });

  it("maps complex types to binary", () => {
    expect(icebergTypeToDataType("list<int>")).toBe("binary");
    expect(icebergTypeToDataType("map<string,int>")).toBe("binary");
    expect(icebergTypeToDataType("struct<a:int>")).toBe("binary");
  });

  it("maps unknown types to binary", () => {
    expect(icebergTypeToDataType("something_unknown")).toBe("binary");
  });
});

describe("parseIcebergMetadata", () => {
  const sampleV2 = JSON.stringify({
    "format-version": 2,
    "table-uuid": "test-uuid",
    "location": "s3://my-bucket/tables/orders",
    "current-snapshot-id": 12345,
    "current-schema-id": 0,
    "schemas": [{
      "schema-id": 0,
      "type": "struct",
      "fields": [
        { "id": 1, "name": "id", "required": true, "type": "long" },
        { "id": 2, "name": "amount", "required": false, "type": "double" },
        { "id": 3, "name": "name", "required": false, "type": "string" },
      ],
    }],
    "snapshots": [{
      "snapshot-id": 12345,
      "timestamp-ms": 1700000000000,
      "manifest-list": "s3://my-bucket/tables/orders/metadata/snap-12345-1-abc.avro",
    }],
  });

  it("parses v2 metadata correctly", () => {
    const result = parseIcebergMetadata(sampleV2);
    expect(result).not.toBe(null);
    expect(result!.currentSnapshotId).toBe("12345");
    expect(result!.schema.fields).toHaveLength(3);
    expect(result!.schema.fields[0].name).toBe("id");
    expect(result!.schema.fields[0].type).toBe("long");
    expect(result!.schema.fields[0].required).toBe(true);
    expect(result!.manifestListPath).toBe("metadata/snap-12345-1-abc.avro");
  });

  it("parses v1 metadata with top-level schema", () => {
    const v1 = JSON.stringify({
      "format-version": 1,
      "location": "s3://bucket/table",
      "current-snapshot-id": 999,
      "schema": {
        "type": "struct",
        "fields": [
          { "id": 1, "name": "col1", "required": true, "type": "int" },
        ],
      },
      "snapshots": [{
        "snapshot-id": 999,
        "manifest-list": "s3://bucket/table/metadata/snap-999.avro",
      }],
    });
    const result = parseIcebergMetadata(v1);
    expect(result).not.toBe(null);
    expect(result!.schema.fields).toHaveLength(1);
    expect(result!.manifestListPath).toBe("metadata/snap-999.avro");
  });

  it("returns null for invalid JSON", () => {
    expect(parseIcebergMetadata("{invalid")).toBe(null);
  });

  it("returns null for missing snapshot", () => {
    const meta = JSON.stringify({
      "current-snapshot-id": 999,
      "schemas": [{ "schema-id": 0, "fields": [{ "id": 1, "name": "x", "type": "int", "required": true }] }],
      "snapshots": [{ "snapshot-id": 888, "manifest-list": "foo.avro" }],
    });
    expect(parseIcebergMetadata(meta)).toBe(null);
  });

  it("returns null for missing fields", () => {
    expect(parseIcebergMetadata("{}")).toBe(null);
    expect(parseIcebergMetadata("null")).toBe(null);
  });
});

describe("extractParquetPathsFromManifest", () => {
  it("extracts .parquet paths from binary content", () => {
    const text = "some-binary-data data/part-00001.parquet more-stuff data/part-00002.parquet end";
    const encoder = new TextEncoder();
    const result = extractParquetPathsFromManifest(encoder.encode(text).buffer);
    expect(result).toContain("data/part-00001.parquet");
    expect(result).toContain("data/part-00002.parquet");
    expect(result).toHaveLength(2);
  });

  it("strips S3 prefixes", () => {
    const text = "s3://my-bucket/tables/orders/data/part-00001.parquet";
    const encoder = new TextEncoder();
    const result = extractParquetPathsFromManifest(encoder.encode(text).buffer);
    expect(result[0]).toBe("tables/orders/data/part-00001.parquet");
  });

  it("deduplicates paths", () => {
    const text = "data/part-00001.parquet data/part-00001.parquet";
    const encoder = new TextEncoder();
    const result = extractParquetPathsFromManifest(encoder.encode(text).buffer);
    expect(result).toHaveLength(1);
  });

  it("returns empty for no matches", () => {
    const encoder = new TextEncoder();
    const result = extractParquetPathsFromManifest(encoder.encode("no parquet files here").buffer);
    expect(result).toEqual([]);
  });
});
