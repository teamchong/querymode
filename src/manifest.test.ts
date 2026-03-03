import { describe, it, expect } from "vitest";
import { parseManifest } from "./manifest.js";
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
});
