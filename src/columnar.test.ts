import { describe, it, expect } from "vitest";
import {
  wasmResultToQMCB,
  decodeColumnarBatch,
  encodeColumnarBatch,
  columnarBatchToRows,
  concatQMCBBatches,
  DTYPE_F64,
  DTYPE_I64,
  DTYPE_I32,
  DTYPE_UTF8,
  DTYPE_BOOL,
  type ColumnarBatch,
} from "./columnar.js";

// ============================================================================
// Helpers: build fake WASM result buffers
// ============================================================================

/**
 * Build a WASM result buffer matching the format that sql_executor.zig writes.
 *
 * Header (28 bytes):
 *   [4] magic (ignored by parser)
 *   [4] numColumns
 *   [4] numRows
 *   [4] ignored
 *   [4] dataStart
 *   [4] ignored
 *   [4] ignored
 *
 * Per column descriptor:
 *   [1] nameLength
 *   [nameLength] name
 *   [1] columnType (0=i64, 1=f64, 2=string, 3=bool, 4=i32, 5=f32)
 *
 * Data: column-major (all values for col0, then all values for col1, ...)
 */
function buildWasmResult(
  columns: { name: string; type: number; values: (number | bigint | string | boolean)[] }[],
): ArrayBuffer {
  const numRows = columns[0]?.values.length ?? 0;
  const numColumns = columns.length;
  const enc = new TextEncoder();

  // Compute descriptor size
  let descSize = 0;
  for (const col of columns) {
    descSize += 1 + enc.encode(col.name).length + 1;
  }
  const dataStart = 28 + descSize;

  // Compute data size
  let dataSize = 0;
  for (const col of columns) {
    switch (col.type) {
      case 0: // i64
      case 1: // f64
        dataSize += numRows * 8;
        break;
      case 4: // i32
      case 5: // f32
        dataSize += numRows * 4;
        break;
      case 3: // bool
        dataSize += numRows;
        break;
      case 2: // string
        for (const v of col.values) {
          const s = v as string;
          dataSize += 4 + enc.encode(s).length;
        }
        break;
    }
  }

  const totalSize = dataStart + dataSize;
  const buf = new ArrayBuffer(totalSize);
  const view = new DataView(buf);
  const u8 = new Uint8Array(buf);

  // Header
  view.setUint32(0, 0, true); // magic
  view.setUint32(4, numColumns, true);
  view.setUint32(8, numRows, true);
  view.setUint32(12, 0, true);
  view.setUint32(16, dataStart, true);
  view.setUint32(20, 0, true);
  view.setUint32(24, 0, true);

  // Column descriptors
  let pos = 28;
  for (const col of columns) {
    const nameBytes = enc.encode(col.name);
    u8[pos++] = nameBytes.length;
    u8.set(nameBytes, pos);
    pos += nameBytes.length;
    u8[pos++] = col.type;
  }

  // Column data
  let dp = dataStart;
  for (const col of columns) {
    switch (col.type) {
      case 0: // i64
        for (const v of col.values) {
          view.setBigInt64(dp, BigInt(v as number | bigint), true);
          dp += 8;
        }
        break;
      case 1: // f64
        for (const v of col.values) {
          view.setFloat64(dp, v as number, true);
          dp += 8;
        }
        break;
      case 4: // i32
        for (const v of col.values) {
          view.setInt32(dp, v as number, true);
          dp += 4;
        }
        break;
      case 5: // f32
        for (const v of col.values) {
          view.setFloat32(dp, v as number, true);
          dp += 4;
        }
        break;
      case 3: // bool
        for (const v of col.values) {
          u8[dp++] = (v as boolean) ? 1 : 0;
        }
        break;
      case 2: // string
        for (const v of col.values) {
          const strBytes = enc.encode(v as string);
          view.setUint32(dp, strBytes.length, true);
          dp += 4;
          u8.set(strBytes, dp);
          dp += strBytes.length;
        }
        break;
    }
  }

  return buf;
}

// ============================================================================
// Tests
// ============================================================================

describe("columnar", () => {
  describe("wasmResultToQMCB", () => {
    it("converts f64 columns", () => {
      const wasm = buildWasmResult([
        { name: "x", type: 1, values: [1.5, 2.5, 3.5] },
        { name: "y", type: 1, values: [10.0, 20.0, 30.0] },
      ]);
      const qmcb = wasmResultToQMCB(wasm, 0, wasm.byteLength);
      expect(qmcb).not.toBeNull();

      const batch = decodeColumnarBatch(qmcb!);
      expect(batch).not.toBeNull();
      expect(batch!.rowCount).toBe(3);
      expect(batch!.columns.length).toBe(2);
      expect(batch!.columns[0].name).toBe("x");
      expect(batch!.columns[0].dtype).toBe(DTYPE_F64);
      expect(new Float64Array(batch!.columns[0].data)).toEqual(new Float64Array([1.5, 2.5, 3.5]));
      expect(new Float64Array(batch!.columns[1].data)).toEqual(new Float64Array([10, 20, 30]));
    });

    it("converts i64 columns", () => {
      const wasm = buildWasmResult([
        { name: "id", type: 0, values: [100n, 200n, 300n] },
      ]);
      const qmcb = wasmResultToQMCB(wasm, 0, wasm.byteLength);
      const batch = decodeColumnarBatch(qmcb!);
      expect(batch!.columns[0].dtype).toBe(DTYPE_I64);
      expect(new BigInt64Array(batch!.columns[0].data)).toEqual(new BigInt64Array([100n, 200n, 300n]));
    });

    it("converts i32 columns", () => {
      const wasm = buildWasmResult([
        { name: "count", type: 4, values: [1, 2, 3, 4] },
      ]);
      const qmcb = wasmResultToQMCB(wasm, 0, wasm.byteLength);
      const batch = decodeColumnarBatch(qmcb!);
      expect(batch!.columns[0].dtype).toBe(DTYPE_I32);
      expect(new Int32Array(batch!.columns[0].data)).toEqual(new Int32Array([1, 2, 3, 4]));
    });

    it("converts string columns", () => {
      const wasm = buildWasmResult([
        { name: "name", type: 2, values: ["alice", "bob", "charlie"] },
      ]);
      const qmcb = wasmResultToQMCB(wasm, 0, wasm.byteLength);
      const batch = decodeColumnarBatch(qmcb!);
      expect(batch!.columns[0].dtype).toBe(DTYPE_UTF8);
      expect(batch!.columns[0].offsets).toBeDefined();
      const offsets = batch!.columns[0].offsets!;
      expect(offsets.length).toBe(4); // numRows + 1
      const data = new Uint8Array(batch!.columns[0].data);
      const dec = new TextDecoder();
      expect(dec.decode(data.subarray(offsets[0], offsets[1]))).toBe("alice");
      expect(dec.decode(data.subarray(offsets[1], offsets[2]))).toBe("bob");
      expect(dec.decode(data.subarray(offsets[2], offsets[3]))).toBe("charlie");
    });

    it("converts bool columns", () => {
      const wasm = buildWasmResult([
        { name: "active", type: 3, values: [true, false, true, false, true, false, true, false, true] },
      ]);
      const qmcb = wasmResultToQMCB(wasm, 0, wasm.byteLength);
      const batch = decodeColumnarBatch(qmcb!);
      expect(batch!.columns[0].dtype).toBe(DTYPE_BOOL);
      // Packed bits: true=1, false=0
      const bits = new Uint8Array(batch!.columns[0].data);
      // bit 0=1, bit 1=0, bit 2=1, bit 3=0, bit 4=1, bit 5=0, bit 6=1, bit 7=0 = 0b01010101 = 0x55
      expect(bits[0]).toBe(0x55);
      // bit 8=1 = 0b00000001 = 0x01
      expect(bits[1]).toBe(0x01);
    });

    it("converts mixed columns", () => {
      const wasm = buildWasmResult([
        { name: "id", type: 0, values: [1n, 2n] },
        { name: "score", type: 1, values: [95.5, 87.3] },
        { name: "name", type: 2, values: ["alice", "bob"] },
        { name: "active", type: 3, values: [true, false] },
      ]);
      const qmcb = wasmResultToQMCB(wasm, 0, wasm.byteLength);
      const batch = decodeColumnarBatch(qmcb!);
      expect(batch!.rowCount).toBe(2);
      expect(batch!.columns.length).toBe(4);
    });

    it("returns null for empty result", () => {
      const wasm = buildWasmResult([]);
      expect(wasmResultToQMCB(wasm, 0, wasm.byteLength)).toBeNull();
    });

    it("returns null for too-small buffer", () => {
      expect(wasmResultToQMCB(new ArrayBuffer(10), 0, 10)).toBeNull();
    });
  });

  describe("columnarBatchToRows", () => {
    it("materializes f64 + string batch to Row[]", () => {
      const wasm = buildWasmResult([
        { name: "score", type: 1, values: [95.5, 87.3, 92.0] },
        { name: "name", type: 2, values: ["alice", "bob", "charlie"] },
      ]);
      const qmcb = wasmResultToQMCB(wasm, 0, wasm.byteLength)!;
      const batch = decodeColumnarBatch(qmcb)!;
      const rows = columnarBatchToRows(batch);
      expect(rows.length).toBe(3);
      expect(rows[0]).toEqual({ score: 95.5, name: "alice" });
      expect(rows[1]).toEqual({ score: 87.3, name: "bob" });
      expect(rows[2]).toEqual({ score: 92.0, name: "charlie" });
    });

    it("materializes i64 to bigint", () => {
      const wasm = buildWasmResult([
        { name: "big", type: 0, values: [9007199254740993n] },
      ]);
      const qmcb = wasmResultToQMCB(wasm, 0, wasm.byteLength)!;
      const batch = decodeColumnarBatch(qmcb)!;
      const rows = columnarBatchToRows(batch);
      expect(rows[0].big).toBe(9007199254740993n);
    });

    it("materializes bool columns", () => {
      const wasm = buildWasmResult([
        { name: "flag", type: 3, values: [true, false, true] },
      ]);
      const qmcb = wasmResultToQMCB(wasm, 0, wasm.byteLength)!;
      const batch = decodeColumnarBatch(qmcb)!;
      const rows = columnarBatchToRows(batch);
      expect(rows[0].flag).toBe(true);
      expect(rows[1].flag).toBe(false);
      expect(rows[2].flag).toBe(true);
    });
  });

  describe("encodeColumnarBatch / decodeColumnarBatch round-trip", () => {
    it("round-trips f64 batch", () => {
      const batch: ColumnarBatch = {
        rowCount: 3,
        columns: [{
          name: "val",
          dtype: DTYPE_F64,
          rowCount: 3,
          data: new Float64Array([1.1, 2.2, 3.3]).buffer,
        }],
      };
      const qmcb = encodeColumnarBatch(batch);
      const decoded = decodeColumnarBatch(qmcb)!;
      expect(decoded.rowCount).toBe(3);
      expect(new Float64Array(decoded.columns[0].data)).toEqual(new Float64Array([1.1, 2.2, 3.3]));
    });

    it("round-trips string batch", () => {
      const offsets = new Uint32Array([0, 5, 8, 15]);
      const enc = new TextEncoder();
      const data = enc.encode("alicebobcharlie");
      const batch: ColumnarBatch = {
        rowCount: 3,
        columns: [{
          name: "name",
          dtype: DTYPE_UTF8,
          rowCount: 3,
          data: data.buffer,
          offsets,
        }],
      };
      const qmcb = encodeColumnarBatch(batch);
      const decoded = decodeColumnarBatch(qmcb)!;
      const rows = columnarBatchToRows(decoded);
      expect(rows[0].name).toBe("alice");
      expect(rows[1].name).toBe("bob");
      expect(rows[2].name).toBe("charlie");
    });

    it("round-trips bool batch", () => {
      const packedBits = new Uint8Array([0b00000101]); // true, false, true
      const batch: ColumnarBatch = {
        rowCount: 3,
        columns: [{
          name: "flag",
          dtype: DTYPE_BOOL,
          rowCount: 3,
          data: packedBits.buffer,
        }],
      };
      const qmcb = encodeColumnarBatch(batch);
      const decoded = decodeColumnarBatch(qmcb)!;
      const rows = columnarBatchToRows(decoded);
      expect(rows[0].flag).toBe(true);
      expect(rows[1].flag).toBe(false);
      expect(rows[2].flag).toBe(true);
    });
  });

  describe("concatQMCBBatches", () => {
    it("concatenates two numeric batches", () => {
      const wasm1 = buildWasmResult([{ name: "x", type: 1, values: [1.0, 2.0] }]);
      const wasm2 = buildWasmResult([{ name: "x", type: 1, values: [3.0, 4.0, 5.0] }]);
      const q1 = wasmResultToQMCB(wasm1, 0, wasm1.byteLength)!;
      const q2 = wasmResultToQMCB(wasm2, 0, wasm2.byteLength)!;

      const merged = concatQMCBBatches([q1, q2])!;
      const batch = decodeColumnarBatch(merged)!;
      expect(batch.rowCount).toBe(5);
      expect(new Float64Array(batch.columns[0].data)).toEqual(new Float64Array([1, 2, 3, 4, 5]));
    });

    it("concatenates string batches", () => {
      const wasm1 = buildWasmResult([{ name: "s", type: 2, values: ["hello", "world"] }]);
      const wasm2 = buildWasmResult([{ name: "s", type: 2, values: ["foo"] }]);
      const q1 = wasmResultToQMCB(wasm1, 0, wasm1.byteLength)!;
      const q2 = wasmResultToQMCB(wasm2, 0, wasm2.byteLength)!;

      const merged = concatQMCBBatches([q1, q2])!;
      const batch = decodeColumnarBatch(merged)!;
      const rows = columnarBatchToRows(batch);
      expect(rows.length).toBe(3);
      expect(rows[0].s).toBe("hello");
      expect(rows[1].s).toBe("world");
      expect(rows[2].s).toBe("foo");
    });

    it("concatenates bool batches", () => {
      const wasm1 = buildWasmResult([{ name: "b", type: 3, values: [true, false] }]);
      const wasm2 = buildWasmResult([{ name: "b", type: 3, values: [false, true, true] }]);
      const q1 = wasmResultToQMCB(wasm1, 0, wasm1.byteLength)!;
      const q2 = wasmResultToQMCB(wasm2, 0, wasm2.byteLength)!;

      const merged = concatQMCBBatches([q1, q2])!;
      const batch = decodeColumnarBatch(merged)!;
      const rows = columnarBatchToRows(batch);
      expect(rows.map(r => r.b)).toEqual([true, false, false, true, true]);
    });

    it("returns single batch unchanged", () => {
      const wasm = buildWasmResult([{ name: "x", type: 1, values: [1.0] }]);
      const q = wasmResultToQMCB(wasm, 0, wasm.byteLength)!;
      const result = concatQMCBBatches([q]);
      expect(result).toBe(q); // same reference
    });

    it("returns null for empty array", () => {
      expect(concatQMCBBatches([])).toBeNull();
    });
  });

  describe("end-to-end: WASM → QMCB → decode → Row[]", () => {
    it("full pipeline with mixed types", () => {
      const wasm = buildWasmResult([
        { name: "id", type: 0, values: [1n, 2n, 3n] },
        { name: "score", type: 1, values: [95.5, 87.3, 92.0] },
        { name: "name", type: 2, values: ["alice", "bob", "charlie"] },
        { name: "active", type: 3, values: [true, false, true] },
        { name: "rank", type: 4, values: [1, 2, 3] },
      ]);
      const qmcb = wasmResultToQMCB(wasm, 0, wasm.byteLength)!;
      const batch = decodeColumnarBatch(qmcb)!;
      const rows = columnarBatchToRows(batch);

      expect(rows.length).toBe(3);
      expect(rows[0]).toEqual({ id: 1n, score: 95.5, name: "alice", active: true, rank: 1 });
      expect(rows[1]).toEqual({ id: 2n, score: 87.3, name: "bob", active: false, rank: 2 });
      expect(rows[2]).toEqual({ id: 3n, score: 92.0, name: "charlie", active: true, rank: 3 });
    });

    it("handles empty strings", () => {
      const wasm = buildWasmResult([
        { name: "s", type: 2, values: ["", "hello", ""] },
      ]);
      const qmcb = wasmResultToQMCB(wasm, 0, wasm.byteLength)!;
      const batch = decodeColumnarBatch(qmcb)!;
      const rows = columnarBatchToRows(batch);
      expect(rows[0].s).toBe("");
      expect(rows[1].s).toBe("hello");
      expect(rows[2].s).toBe("");
    });

    it("handles large batch (100k rows)", () => {
      const n = 100_000;
      const values = new Array(n);
      for (let i = 0; i < n; i++) values[i] = i * 1.5;

      const wasm = buildWasmResult([{ name: "x", type: 1, values }]);
      const qmcb = wasmResultToQMCB(wasm, 0, wasm.byteLength)!;
      const batch = decodeColumnarBatch(qmcb)!;
      expect(batch.rowCount).toBe(n);

      const arr = new Float64Array(batch.columns[0].data);
      expect(arr[0]).toBe(0);
      expect(arr[n - 1]).toBe((n - 1) * 1.5);
    });
  });
});
