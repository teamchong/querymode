/**
 * Arrow IPC Reader — minimal implementation for Arrow IPC (File) format.
 *
 * Arrow IPC File format:
 *   MAGIC ("ARROW1")  6 bytes
 *   padding            2 bytes (to 8-byte alignment)
 *   Schema message     (flatbuffer)
 *   Record batches     (flatbuffer metadata + body buffers)
 *   Footer             (flatbuffer)
 *   Footer length      4 bytes (LE i32)
 *   MAGIC ("ARROW1")   6 bytes
 *
 * This reader implements just enough to:
 *   - Detect Arrow IPC files via magic bytes
 *   - Parse the schema from the footer
 *   - Read record batches into ColumnMeta pages
 *   - Create FragmentSources that yield decoded column data
 *
 * No external dependencies. Everything is parsed from raw bytes.
 */

import type { FormatReader, DataSource } from "../reader.js";
import type { ColumnMeta, DataType, PageInfo, Row } from "../types.js";
import type { FragmentSource } from "../operators.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ARROW_MAGIC = new Uint8Array([0x41, 0x52, 0x52, 0x4f, 0x57, 0x31]); // "ARROW1"
const ARROW_MAGIC_SIZE = 6;
const FOOTER_LENGTH_SIZE = 4;
const CONTINUATION_MARKER = 0xffffffff;

// ---------------------------------------------------------------------------
// Flatbuffer helpers — minimal hand-rolled reader
// ---------------------------------------------------------------------------

/** Read a little-endian i32 from a DataView. */
function readI32(view: DataView, offset: number): number {
  return view.getInt32(offset, true);
}

/** Read a little-endian i64 from a DataView as a Number (loses precision >2^53). */
function readI64AsNumber(view: DataView, offset: number): number {
  const lo = view.getUint32(offset, true);
  const hi = view.getInt32(offset + 4, true);
  return hi * 0x100000000 + lo;
}

/** Read a flatbuffer offset (indirect pointer). */
function fbOffset(buf: DataView, tableOffset: number, fieldIdx: number): number {
  const vtable = tableOffset - readI32(buf, tableOffset);
  const vtableLen = buf.getInt16(vtable, true);
  const fieldVtableOffset = 4 + fieldIdx * 2;
  if (fieldVtableOffset >= vtableLen) return 0;
  const fieldOff = buf.getInt16(vtable + fieldVtableOffset, true);
  if (fieldOff === 0) return 0;
  return tableOffset + fieldOff;
}

const textDecoder = new TextDecoder();

/** Read a flatbuffer string. */
function fbString(buf: DataView, strOffset: number): string {
  const len = readI32(buf, strOffset);
  const start = strOffset + 4;
  const bytes = new Uint8Array(buf.buffer, buf.byteOffset + start, len);
  return textDecoder.decode(bytes);
}

/** Read a flatbuffer vector length. */
function fbVectorLen(buf: DataView, vecOffset: number): number {
  return readI32(buf, vecOffset);
}

// ---------------------------------------------------------------------------
// Arrow Schema Parsing
// ---------------------------------------------------------------------------

/** Arrow type IDs from the flatbuffer Type union. */
const enum ArrowTypeId {
  Null = 0,
  Int = 1,
  FloatingPoint = 2,
  Binary = 3,
  Utf8 = 4,
  Bool = 5,
  // Many more exist but we handle the common ones
  FixedSizeBinary = 15,
  LargeUtf8 = 20,
  LargeBinary = 21,
}

interface ArrowField {
  name: string;
  dtype: DataType;
  nullable: boolean;
}

/** Map Arrow type union to our DataType. */
function parseArrowType(buf: DataView, fieldTable: number): DataType {
  // Field: type_type (field idx=2) and type (field idx=3)
  const typeTypeOff = fbOffset(buf, fieldTable, 2);
  const typeType = typeTypeOff ? buf.getUint8(typeTypeOff) : 0;

  const typeOff = fbOffset(buf, fieldTable, 3);
  const typeTable = typeOff ? typeOff + readI32(buf, typeOff) : 0;

  switch (typeType) {
    case ArrowTypeId.Int: {
      if (!typeTable) return "int32";
      // Int type: bitWidth (field 0), is_signed (field 1)
      const bwOff = fbOffset(buf, typeTable, 0);
      const bitWidth = bwOff ? readI32(buf, bwOff) : 32;
      const signedOff = fbOffset(buf, typeTable, 1);
      const isSigned = signedOff ? buf.getUint8(signedOff) !== 0 : true;
      switch (bitWidth) {
        case 8: return isSigned ? "int8" : "uint8";
        case 16: return isSigned ? "int16" : "uint16";
        case 32: return isSigned ? "int32" : "uint32";
        case 64: return isSigned ? "int64" : "uint64";
        default: return "int32";
      }
    }
    case ArrowTypeId.FloatingPoint: {
      if (!typeTable) return "float64";
      // FloatingPoint type: precision (field 0) — 0=HALF, 1=SINGLE, 2=DOUBLE
      const precOff = fbOffset(buf, typeTable, 0);
      const precision = precOff ? readI32(buf, precOff) : 2;
      switch (precision) {
        case 0: return "float16";
        case 1: return "float32";
        case 2: return "float64";
        default: return "float64";
      }
    }
    case ArrowTypeId.Utf8:
    case ArrowTypeId.LargeUtf8:
      return "utf8";
    case ArrowTypeId.Binary:
    case ArrowTypeId.LargeBinary:
    case ArrowTypeId.FixedSizeBinary:
      return "binary";
    case ArrowTypeId.Bool:
      return "bool";
    default:
      return "utf8"; // fallback
  }
}

function parseField(buf: DataView, fieldTable: number): ArrowField {
  // Field flatbuffer: name (field 0), nullable (field 1), type_type (field 2), type (field 3)
  const nameOff = fbOffset(buf, fieldTable, 0);
  const name = nameOff ? fbString(buf, nameOff + readI32(buf, nameOff)) : "";
  const nullableOff = fbOffset(buf, fieldTable, 1);
  const nullable = nullableOff ? buf.getUint8(nullableOff) !== 0 : true;
  const dtype = parseArrowType(buf, fieldTable);
  return { name, dtype, nullable };
}

function parseSchema(buf: DataView, schemaTable: number): ArrowField[] {
  // Schema flatbuffer: fields (field 0)
  const fieldsOff = fbOffset(buf, schemaTable, 0);
  if (!fieldsOff) return [];
  const fieldsVec = fieldsOff + readI32(buf, fieldsOff);
  const numFields = fbVectorLen(buf, fieldsVec);
  const fields: ArrowField[] = [];
  for (let i = 0; i < numFields; i++) {
    const elemOff = fieldsVec + 4 + i * 4;
    const fieldTable = elemOff + readI32(buf, elemOff);
    fields.push(parseField(buf, fieldTable));
  }
  return fields;
}

// ---------------------------------------------------------------------------
// Footer Parsing
// ---------------------------------------------------------------------------

interface RecordBatchBlock {
  offset: number;  // byte offset from start of file
  metaDataLength: number;
  bodyLength: number;
}

interface ArrowFooter {
  schema: ArrowField[];
  recordBatches: RecordBatchBlock[];
}

function parseFooter(buf: DataView, footerTable: number): ArrowFooter {
  // Footer flatbuffer: schema (field 1), recordBatches (field 3)
  const schemaOff = fbOffset(buf, footerTable, 1);
  const schema = schemaOff ? parseSchema(buf, schemaOff + readI32(buf, schemaOff)) : [];

  const rbOff = fbOffset(buf, footerTable, 3);
  const recordBatches: RecordBatchBlock[] = [];
  if (rbOff) {
    const rbVec = rbOff + readI32(buf, rbOff);
    const numRb = fbVectorLen(buf, rbVec);
    // Record batch blocks are inline structs: (i64 offset, i32 metaDataLength, i64 bodyLength)
    // Total struct size = 24 bytes
    const structStart = rbVec + 4;
    for (let i = 0; i < numRb; i++) {
      const base = structStart + i * 24;
      const offset = readI64AsNumber(buf, base);
      const metaDataLength = readI32(buf, base + 8);
      const bodyLength = readI64AsNumber(buf, base + 16);
      recordBatches.push({ offset, metaDataLength, bodyLength });
    }
  }

  return { schema, recordBatches };
}

// ---------------------------------------------------------------------------
// RecordBatch Metadata Parsing
// ---------------------------------------------------------------------------

interface FieldNode {
  length: number;
  nullCount: number;
}

interface BufferDescriptor {
  offset: number;
  length: number;
}

interface RecordBatchMeta {
  length: number;
  nodes: FieldNode[];
  buffers: BufferDescriptor[];
}

function parseRecordBatchMessage(buf: DataView, msgTable: number): RecordBatchMeta | null {
  // Message flatbuffer: header_type (field 1), header (field 2)
  const headerTypeOff = fbOffset(buf, msgTable, 1);
  const headerType = headerTypeOff ? buf.getUint8(headerTypeOff) : 0;
  if (headerType !== 3) return null; // 3 = RecordBatch

  const headerOff = fbOffset(buf, msgTable, 2);
  if (!headerOff) return null;
  const rbTable = headerOff + readI32(buf, headerOff);

  // RecordBatch flatbuffer: length (field 0), nodes (field 1), buffers (field 2)
  const lengthOff = fbOffset(buf, rbTable, 0);
  const length = lengthOff ? readI64AsNumber(buf, lengthOff) : 0;

  const nodesOff = fbOffset(buf, rbTable, 1);
  const nodes: FieldNode[] = [];
  if (nodesOff) {
    const nodesVec = nodesOff + readI32(buf, nodesOff);
    const numNodes = fbVectorLen(buf, nodesVec);
    // FieldNode is an inline struct: (i64 length, i64 null_count) = 16 bytes
    const nodeStart = nodesVec + 4;
    for (let i = 0; i < numNodes; i++) {
      const base = nodeStart + i * 16;
      nodes.push({
        length: readI64AsNumber(buf, base),
        nullCount: readI64AsNumber(buf, base + 8),
      });
    }
  }

  const bufsOff = fbOffset(buf, rbTable, 2);
  const buffers: BufferDescriptor[] = [];
  if (bufsOff) {
    const bufsVec = bufsOff + readI32(buf, bufsOff);
    const numBufs = fbVectorLen(buf, bufsVec);
    // Buffer is an inline struct: (i64 offset, i64 length) = 16 bytes
    const bufStart = bufsVec + 4;
    for (let i = 0; i < numBufs; i++) {
      const base = bufStart + i * 16;
      buffers.push({
        offset: readI64AsNumber(buf, base),
        length: readI64AsNumber(buf, base + 8),
      });
    }
  }

  return { length, nodes, buffers };
}

/** Check whether a DataType uses variable-length encoding (offsets buffer). */
function isVariableLength(dtype: DataType): boolean {
  return dtype === "utf8" || dtype === "binary";
}

// ---------------------------------------------------------------------------
// ArrowFragmentSource
// ---------------------------------------------------------------------------

class ArrowFragmentSource implements FragmentSource {
  columns: ColumnMeta[];
  private source: DataSource;

  constructor(columns: ColumnMeta[], source: DataSource) {
    this.columns = columns;
    this.source = source;
  }

  /**
   * readPage reads raw column data for a page from the Arrow IPC file.
   *
   * Pages in Arrow map to record batch buffers. The byteOffset and byteLength
   * on each PageInfo point directly to the data buffer region within the file.
   *
   * For fixed-width types we return the raw buffer and let decodePage() handle it.
   * For variable-length types (utf8, binary) we re-encode into the length-prefixed
   * wire format that decodePage() expects (u32-len + bytes per value).
   */
  async readPage(col: ColumnMeta, page: PageInfo): Promise<ArrayBuffer> {
    const raw = await this.source.readRange(Number(page.byteOffset), page.byteLength);

    if (!isVariableLength(col.dtype)) {
      return raw;
    }

    // Variable-length: `raw` is the data buffer. The offsets buffer was stored
    // at an adjacent page. We need to decode the offsets to split the data.
    // Our page encoding stores the offsets buffer info in the encoding field.
    if (page.encoding && page.encoding.dictionaryPageOffset !== undefined && page.encoding.dictionaryPageLength) {
      const offsetsBuf = await this.source.readRange(
        Number(page.encoding.dictionaryPageOffset),
        page.encoding.dictionaryPageLength,
      );
      return reencodeVarLen(offsetsBuf, raw, page.rowCount);
    }

    // Fallback: treat the entire buffer as a single string
    return raw;
  }
}

/**
 * Re-encode a variable-length column from Arrow (offsets + data) into the
 * length-prefixed format used by decodePage(): [u32 len][bytes]...
 *
 * Supports both regular (32-bit) and Large (64-bit) offset types by detecting
 * the offset size from the buffer length relative to rowCount.
 */
function reencodeVarLen(
  offsetsBuf: ArrayBuffer,
  dataBuf: ArrayBuffer,
  rowCount: number,
): ArrayBuffer {
  const offsetsView = new DataView(offsetsBuf);
  const dataBytes = new Uint8Array(dataBuf);

  // Detect 64-bit (Large) offsets: LargeUtf8/LargeBinary use i64 offsets
  const isLargeOffsets = offsetsBuf.byteLength >= (rowCount + 1) * 8;
  const offsetSize = isLargeOffsets ? 8 : 4;
  const numOffsets = Math.min(rowCount + 1, Math.floor(offsetsBuf.byteLength / offsetSize));
  if (numOffsets < 2) return new ArrayBuffer(0);

  const readOffset = isLargeOffsets
    ? (i: number) => readI64AsNumber(offsetsView, i * 8)
    : (i: number) => offsetsView.getInt32(i * 4, true);

  // First pass: compute total output size
  let totalSize = 0;
  const count = numOffsets - 1;
  for (let i = 0; i < count; i++) {
    const start = readOffset(i);
    const end = readOffset(i + 1);
    totalSize += 4 + (end - start);
  }

  // Second pass: write length-prefixed values
  const out = new Uint8Array(totalSize);
  const outView = new DataView(out.buffer);
  let pos = 0;
  for (let i = 0; i < count; i++) {
    const start = readOffset(i);
    const end = readOffset(i + 1);
    const len = end - start;
    outView.setUint32(pos, len, true);
    pos += 4;
    if (len > 0) {
      out.set(dataBytes.subarray(start, end), pos);
      pos += len;
    }
  }

  return out.buffer;
}

// ---------------------------------------------------------------------------
// ArrowReader — implements FormatReader
// ---------------------------------------------------------------------------

export class ArrowReader implements FormatReader {
  extensions = [".arrow", ".ipc", ".feather"];

  /** Arrow IPC files end with "ARROW1" magic (last 6 bytes). */
  canRead(tailBytes: ArrayBuffer): boolean {
    const tail = new Uint8Array(tailBytes);
    if (tail.length < ARROW_MAGIC_SIZE) return false;
    const magicStart = tail.length - ARROW_MAGIC_SIZE;
    for (let i = 0; i < ARROW_MAGIC_SIZE; i++) {
      if (tail[magicStart + i] !== ARROW_MAGIC[i]) return false;
    }
    return true;
  }

  async readMeta(source: DataSource): Promise<{ columns: ColumnMeta[]; totalRows: number }> {
    const footer = await this.readFooter(source);
    const columns = this.buildColumnMeta(footer, source);
    // To get accurate row counts we need to parse record batch messages
    const totalRows = await this.computeTotalRows(source, footer);
    // Update page row counts
    await this.populatePages(columns, source, footer);
    return { columns, totalRows };
  }

  async createFragments(source: DataSource, projected: ColumnMeta[]): Promise<FragmentSource[]> {
    return [new ArrowFragmentSource(projected, source)];
  }

  // -------------------------------------------------------------------------
  // Internal
  // -------------------------------------------------------------------------

  private async readFooter(source: DataSource): Promise<ArrowFooter> {
    const fileSize = await source.getSize();
    // Read the last 10 bytes: footer_length (4) + magic (6)
    const tailBuf = await source.readRange(fileSize - FOOTER_LENGTH_SIZE - ARROW_MAGIC_SIZE, FOOTER_LENGTH_SIZE + ARROW_MAGIC_SIZE);
    const tailView = new DataView(tailBuf);
    const footerLength = readI32(tailView, 0);

    // Read the footer flatbuffer
    const footerStart = fileSize - FOOTER_LENGTH_SIZE - ARROW_MAGIC_SIZE - footerLength;
    const footerBuf = await source.readRange(footerStart, footerLength);
    const footerView = new DataView(footerBuf);

    // The footer flatbuffer root table offset is at byte 0
    const rootOffset = readI32(footerView, 0);
    return parseFooter(footerView, rootOffset);
  }

  private buildColumnMeta(footer: ArrowFooter, _source: DataSource): ColumnMeta[] {
    return footer.schema.map(field => ({
      name: field.name,
      dtype: field.dtype,
      pages: [], // populated later by populatePages
      nullCount: 0,
    }));
  }

  private async computeTotalRows(source: DataSource, footer: ArrowFooter): Promise<number> {
    let total = 0;
    for (const block of footer.recordBatches) {
      const meta = await this.readRecordBatchMeta(source, block);
      if (meta) total += meta.length;
    }
    return total;
  }

  private async populatePages(
    columns: ColumnMeta[],
    source: DataSource,
    footer: ArrowFooter,
  ): Promise<void> {
    for (const block of footer.recordBatches) {
      const meta = await this.readRecordBatchMeta(source, block);
      if (!meta) continue;

      // Body starts right after the metadata in the file
      const bodyOffset = block.offset + block.metaDataLength;

      // Each field gets 1-2 buffers in the record batch:
      //   - nullable: validity bitmap + data (or validity + offsets + data for var-len)
      //   - non-nullable: data only (or offsets + data for var-len)
      // The buffer ordering matches field order in the schema.
      let bufIdx = 0;
      for (let fi = 0; fi < columns.length && fi < meta.nodes.length; fi++) {
        const col = columns[fi];
        const node = meta.nodes[fi];
        const nullable = footer.schema[fi]?.nullable ?? true;

        // Validity bitmap buffer (always present in IPC even if non-nullable)
        const validityBuf = meta.buffers[bufIdx++];

        if (isVariableLength(col.dtype)) {
          // Offsets buffer
          const offsetsBuf = meta.buffers[bufIdx++];
          // Data buffer
          const dataBuf = meta.buffers[bufIdx++];
          if (dataBuf && offsetsBuf) {
            // nullCount must be 0: readPage re-encodes via reencodeVarLen without bitmap.
            // decodePage would corrupt data if it tried to strip a nonexistent bitmap.
            const page: PageInfo = {
              byteOffset: BigInt(bodyOffset + dataBuf.offset),
              byteLength: dataBuf.length,
              rowCount: node.length,
              nullCount: 0,
              // Store offsets buffer location in encoding so ArrowFragmentSource can retrieve it
              encoding: {
                dictionaryPageOffset: BigInt(bodyOffset + offsetsBuf.offset),
                dictionaryPageLength: offsetsBuf.length,
              },
            };
            col.pages.push(page);
            col.nullCount += node.nullCount;
          }
        } else {
          // Fixed-width: just data buffer
          const dataBuf = meta.buffers[bufIdx++];
          if (dataBuf) {
            // nullCount must be 0: readPage returns raw data buffer without bitmap.
            // decodePage would corrupt data if it tried to strip a nonexistent bitmap.
            const page: PageInfo = {
              byteOffset: BigInt(bodyOffset + dataBuf.offset),
              byteLength: dataBuf.length,
              rowCount: node.length,
              nullCount: 0,
            };
            col.pages.push(page);
            col.nullCount += node.nullCount;
          }
        }
      }
    }
  }

  private async readRecordBatchMeta(
    source: DataSource,
    block: RecordBatchBlock,
  ): Promise<RecordBatchMeta | null> {
    // Read the message metadata from the record batch block
    const metaBuf = await source.readRange(block.offset, block.metaDataLength);
    const metaView = new DataView(metaBuf);

    let offset = 0;
    // Check for continuation marker (0xFFFFFFFF)
    if (metaView.getUint32(0, true) === CONTINUATION_MARKER) {
      offset += 4;
      // Next 4 bytes = flatbuffer size (which we already know from metaDataLength)
      offset += 4;
    }

    // The message flatbuffer root table
    const rootOffset = offset + readI32(metaView, offset);
    return parseRecordBatchMessage(metaView, rootOffset);
  }
}
