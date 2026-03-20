import { readFileSync, existsSync } from "node:fs";
import { describe, it, expect } from "vitest";
import { parseManifest, buildManifestBinary, schemaFromColumns } from "./manifest.js";
import { LANCE_MAGIC } from "./footer.js";

/** Encode a number as a protobuf varint */
function encodeVarint(value: number): number[] {
  const bytes: number[] = [];
  while (value > 0x7f) {
    bytes.push((value & 0x7f) | 0x80);
    value >>>= 7;
  }
  bytes.push(value & 0x7f);
  return bytes;
}

/** Build a DataFile sub-message with path string */
function buildDataFile(path: string): number[] {
  const pathBytes = new TextEncoder().encode(path);
  // Sub-sub-field 1 length-delimited: tag = 0x0a
  return [0x0a, ...encodeVarint(pathBytes.length), ...pathBytes];
}

/** Build a DataFragment sub-message */
function buildFragment(id: number, path: string, rows: number): number[] {
  const dataFile = buildDataFile(path);
  const parts: number[] = [];
  // Sub-field 1 varint (id): tag = 0x08
  parts.push(0x08, ...encodeVarint(id));
  // Sub-field 2 length-delimited (DataFile): tag = 0x12
  parts.push(0x12, ...encodeVarint(dataFile.length), ...dataFile);
  // Sub-field 4 varint (physical_rows): tag = 0x20
  parts.push(0x20, ...encodeVarint(rows));
  return parts;
}

/** Build a complete manifest binary buffer */
function buildManifestBuf(fragments: { id: number; path: string; rows: number }[], version: number): ArrayBuffer {
  const proto: number[] = [];

  for (const f of fragments) {
    const frag = buildFragment(f.id, f.path, f.rows);
    // Field 2 length-delimited (DataFragment): tag = 0x12
    proto.push(0x12, ...encodeVarint(frag.length), ...frag);
  }

  // Field 3 varint (version): tag = 0x18
  proto.push(0x18, ...encodeVarint(version));

  const protoLen = proto.length;
  const totalSize = 4 + protoLen + 12 + 4; // len + proto + padding + magic
  const buf = new ArrayBuffer(totalSize);
  const view = new DataView(buf);
  const bytes = new Uint8Array(buf);

  view.setUint32(0, protoLen, true);
  bytes.set(proto, 4);
  // 12 bytes of zero padding are already zero
  view.setUint32(4 + protoLen + 12, LANCE_MAGIC, true);

  return buf;
}

describe("parseManifest", () => {
  it("parses manifest with 2 fragments", () => {
    const buf = buildManifestBuf([
      { id: 1, path: "data/0.lance", rows: 500 },
      { id: 2, path: "data/1.lance", rows: 300 },
    ], 7);

    const manifest = parseManifest(buf);
    expect(manifest).not.toBeNull();
    expect(manifest!.version).toBe(7);
    expect(manifest!.fragments).toHaveLength(2);
    expect(manifest!.fragments[0].id).toBe(1);
    expect(manifest!.fragments[0].filePath).toBe("data/0.lance");
    expect(manifest!.fragments[0].physicalRows).toBe(500);
    expect(manifest!.fragments[1].id).toBe(2);
    expect(manifest!.fragments[1].filePath).toBe("data/1.lance");
    expect(manifest!.fragments[1].physicalRows).toBe(300);
    expect(manifest!.totalRows).toBe(800);
  });

  it("returns null for invalid magic", () => {
    const buf = buildManifestBuf([{ id: 1, path: "x.lance", rows: 10 }], 1);
    const view = new DataView(buf);
    // Overwrite magic with garbage
    const protoLen = view.getUint32(0, true);
    view.setUint32(4 + protoLen + 12, 0xdeadbeef, true);
    expect(parseManifest(buf)).toBeNull();
  });

  it("returns null for too-small buffer", () => {
    expect(parseManifest(new ArrayBuffer(8))).toBeNull();
  });

  it("handles manifest with 0 fragments", () => {
    const buf = buildManifestBuf([], 3);
    const manifest = parseManifest(buf);
    expect(manifest).not.toBeNull();
    expect(manifest!.version).toBe(3);
    expect(manifest!.fragments).toHaveLength(0);
    expect(manifest!.totalRows).toBe(0);
  });

  it("parses schema from existing manifests", () => {
    const buf = buildManifestBuf([], 3);
    const manifest = parseManifest(buf);
    expect(manifest!.schema).toEqual([]);
  });
});

describe("buildManifestBinary round-trip", () => {
  it("buildManifestBinary output is parseable by parseManifest", () => {
    const fragments = [
      { id: 1, filePath: "data/abc.lance", physicalRows: 1000 },
      { id: 2, filePath: "data/def.lance", physicalRows: 500 },
    ];
    const buf = buildManifestBinary(5, fragments);
    const manifest = parseManifest(buf);
    expect(manifest).not.toBeNull();
    expect(manifest!.version).toBe(5);
    expect(manifest!.fragments).toHaveLength(2);
    expect(manifest!.fragments[0].id).toBe(1);
    expect(manifest!.fragments[0].filePath).toBe("data/abc.lance");
    expect(manifest!.fragments[0].physicalRows).toBe(1000);
    expect(manifest!.fragments[1].id).toBe(2);
    expect(manifest!.fragments[1].filePath).toBe("data/def.lance");
    expect(manifest!.fragments[1].physicalRows).toBe(500);
    expect(manifest!.totalRows).toBe(1500);
  });

  it("round-trips empty manifest", () => {
    const buf = buildManifestBinary(1, []);
    const manifest = parseManifest(buf);
    expect(manifest).not.toBeNull();
    expect(manifest!.version).toBe(1);
    expect(manifest!.fragments).toHaveLength(0);
  });

  it("round-trips schema fields", () => {
    const schema = schemaFromColumns([
      { name: "id", dtype: "int64" },
      { name: "value", dtype: "float64" },
      { name: "name", dtype: "utf8" },
      { name: "active", dtype: "bool" },
    ]);
    const fragments = [
      { id: 1, filePath: "data/test.lance", physicalRows: 100 },
    ];
    const buf = buildManifestBinary(2, fragments, schema);
    const manifest = parseManifest(buf);
    expect(manifest).not.toBeNull();
    expect(manifest!.version).toBe(2);
    expect(manifest!.schema).toHaveLength(4);
    expect(manifest!.schema[0].name).toBe("id");
    expect(manifest!.schema[0].logicalType).toBe("int64");
    expect(manifest!.schema[0].id).toBe(1);
    expect(manifest!.schema[1].name).toBe("value");
    expect(manifest!.schema[1].logicalType).toBe("double");
    expect(manifest!.schema[2].name).toBe("name");
    expect(manifest!.schema[2].logicalType).toBe("utf8");
    expect(manifest!.schema[3].name).toBe("active");
    expect(manifest!.schema[3].logicalType).toBe("boolean");
  });

  it("round-trips nullable schema fields", () => {
    const schema = [
      { name: "col", logicalType: "int64", id: 1, parentId: 0, nullable: true },
    ];
    const buf = buildManifestBinary(1, [], schema);
    const manifest = parseManifest(buf);
    expect(manifest!.schema).toHaveLength(1);
    expect(manifest!.schema[0].nullable).toBe(true);
  });
});

describe.skipIf(!existsSync("wasm/tests/fixtures/simple_int64.lance/_versions/1.manifest"))("parseManifest fixtures", () => {
  it("parses simple_int64 manifest with schema", () => {
    const buf = readFileSync("wasm/tests/fixtures/simple_int64.lance/_versions/1.manifest");
    const manifest = parseManifest(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    expect(manifest).not.toBeNull();
    expect(manifest!.version).toBe(1);
    expect(manifest!.fragments).toHaveLength(1);
    expect(manifest!.schema).toHaveLength(1);
    expect(manifest!.schema[0].name).toBe("id");
    expect(manifest!.schema[0].logicalType).toBe("int64");
  });

  it("parses mixed_types manifest with multiple schema fields", () => {
    const buf = readFileSync("wasm/tests/fixtures/mixed_types.lance/_versions/1.manifest");
    const manifest = parseManifest(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    expect(manifest).not.toBeNull();
    expect(manifest!.schema.length).toBeGreaterThanOrEqual(3);
    const names = manifest!.schema.map(f => f.name);
    expect(names).toContain("id");
    expect(names).toContain("value");
    expect(names).toContain("name");
    const valueField = manifest!.schema.find(f => f.name === "value");
    expect(valueField!.logicalType).toBe("double");
    const nameField = manifest!.schema.find(f => f.name === "name");
    expect(nameField!.logicalType).toBe("string");
  });

  it("parses simple_float64 manifest", () => {
    const buf = readFileSync("wasm/tests/fixtures/simple_float64.lance/_versions/1.manifest");
    const manifest = parseManifest(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
    expect(manifest).not.toBeNull();
    expect(manifest!.schema).toHaveLength(1);
    expect(manifest!.schema[0].name).toBe("value");
    expect(manifest!.schema[0].logicalType).toBe("double");
  });
});
