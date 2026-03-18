import type { DataType, IcebergSchema } from "./types.js";

const icebergTypeMap: Record<string, DataType> = {
  boolean: "bool", int: "int32", long: "int64", float: "float32", double: "float64",
  string: "utf8", binary: "binary", decimal: "float64", date: "int32",
  time: "int64", timestamp: "int64", timestamptz: "int64", uuid: "utf8", fixed: "binary",
};

/** Map Iceberg type strings to QueryMode DataType. */
export function icebergTypeToDataType(iceType: string): DataType {
  return icebergTypeMap[iceType] ?? "binary";
}

/**
 * Parse Iceberg table metadata JSON into schema, snapshot ID, and manifest list path.
 * Returns null if any critical field is missing or the JSON is invalid.
 */
export function parseIcebergMetadata(json: string): {
  schema: IcebergSchema;
  currentSnapshotId: string;
  manifestListPath: string;
} | null {
  let meta: Record<string, unknown>;
  try {
    meta = JSON.parse(json);
  } catch {
    return null;
  }

  if (typeof meta !== "object" || meta === null) return null;

  // Extract current-snapshot-id
  const currentSnapshotId = meta["current-snapshot-id"];
  if (currentSnapshotId == null) return null;
  const snapshotIdStr = String(currentSnapshotId);

  // Extract schema fields
  let fields: { id: number; name: string; required: boolean; type: string }[] | undefined;

  const currentSchemaId = meta["current-schema-id"];
  const schemas = meta["schemas"] as
    | { "schema-id": number; fields: typeof fields }[]
    | undefined;

  if (Array.isArray(schemas) && schemas.length > 0) {
    // Format v2: find schema matching current-schema-id, or fall back to last
    const matched = schemas.find(
      (s) => s["schema-id"] === currentSchemaId
    );
    fields = (matched ?? schemas[schemas.length - 1]).fields;
  } else {
    // Format v1: top-level "schema" object
    const topSchema = meta["schema"] as
      | { fields: typeof fields }
      | undefined;
    if (topSchema && Array.isArray(topSchema.fields)) {
      fields = topSchema.fields;
    }
  }

  if (!Array.isArray(fields) || fields.length === 0) return null;

  const schema: IcebergSchema = {
    fields: fields
      .filter((f) => f.name && f.type)
      .map((f) => ({
        name: f.name,
        type: f.type,
        required: !!f.required,
      })),
  };
  if (schema.fields.length === 0) return null;

  // Find the current snapshot and its manifest-list
  const snapshots = meta["snapshots"] as
    | { "snapshot-id": number | string; "manifest-list": string }[]
    | undefined;
  if (!Array.isArray(snapshots) || snapshots.length === 0) return null;

  const currentSnapshot = snapshots.find(
    (s) => String(s["snapshot-id"]) === snapshotIdStr
  );
  if (!currentSnapshot) return null;

  const manifestListFull = currentSnapshot["manifest-list"];
  if (typeof manifestListFull !== "string" || manifestListFull.length === 0)
    return null;

  // Extract relative path by stripping the table location prefix
  let manifestListPath = manifestListFull;
  const location = meta["location"];
  if (typeof location === "string" && manifestListFull.startsWith(location)) {
    manifestListPath = manifestListFull.slice(location.length);
    // Strip leading slash
    if (manifestListPath.startsWith("/")) {
      manifestListPath = manifestListPath.slice(1);
    }
  } else {
    // Fallback: extract metadata/snap-*.avro portion
    const metadataIdx = manifestListFull.indexOf("metadata/");
    if (metadataIdx !== -1) {
      manifestListPath = manifestListFull.slice(metadataIdx);
    }
  }

  return { schema, currentSnapshotId: snapshotIdStr, manifestListPath };
}

/**
 * Extract Parquet file paths from an Iceberg manifest (Avro binary).
 * Uses string scanning as a pragmatic initial approach — the Avro container
 * format embeds file paths as length-prefixed strings which are readable as text.
 */
export function extractParquetPathsFromManifest(
  manifestBytes: ArrayBuffer
): string[] {
  const decoder = new TextDecoder();
  const text = decoder.decode(manifestBytes);

  // Match paths ending in .parquet — captures S3/R2 URLs or relative paths
  const parquetPattern = /[a-zA-Z0-9_\-/.:%]+\.parquet/g;
  const matches = text.match(parquetPattern);
  if (!matches) return [];

  const seen = new Set<string>();
  const paths: string[] = [];

  for (const raw of matches) {
    // Strip S3/R2 prefix if present (s3://bucket/...)
    let path = raw;
    const s3Match = path.match(/^s3:\/\/[^/]+\/(.+)$/);
    if (s3Match) {
      path = s3Match[1];
    }

    if (!seen.has(path)) {
      seen.add(path);
      paths.push(path);
    }
  }

  return paths;
}
