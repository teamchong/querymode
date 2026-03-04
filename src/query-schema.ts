/**
 * Zod schemas for query descriptor validation.
 *
 * Validates POST bodies at the API boundary (query-do.ts) before they
 * reach the execution engine. Catches malformed filters, invalid ops,
 * bad aggregate fns, etc. with clear error messages.
 */
import { z } from "zod/v4";

const filterOpSchema = z.object({
  column: z.string().min(1, "Filter column name cannot be empty"),
  op: z.enum(["eq", "neq", "gt", "gte", "lt", "lte", "in"]),
  value: z.union([
    z.number(),
    z.string(),
    z.array(z.union([z.number(), z.string()])),
  ]),
});

const aggregateOpSchema = z.object({
  fn: z.enum(["sum", "avg", "min", "max", "count"]),
  column: z.string().min(1, "Aggregate column name cannot be empty"),
  alias: z.string().optional(),
});

const vectorSearchSchema = z.object({
  column: z.string().min(1),
  queryVector: z.union([
    z.array(z.number()),
    z.instanceof(Float32Array),
  ]),
  topK: z.number().int().positive(),
});

export const queryDescriptorSchema = z.object({
  table: z.string().min(1, "Table name is required"),
  filters: z.array(filterOpSchema).default([]),
  projections: z.array(z.string()).default([]),
  select: z.array(z.string()).optional(), // alias for projections
  sortColumn: z.string().optional(),
  sortDirection: z.enum(["asc", "desc"]).optional(),
  limit: z.number().int().positive().optional(),
  offset: z.number().int().nonnegative().optional(),
  vectorSearch: vectorSearchSchema.optional(),
  aggregates: z.array(aggregateOpSchema).optional(),
  groupBy: z.array(z.string()).optional(),
  cacheTTL: z.number().int().positive().optional(),
});

export type ValidatedQuery = z.infer<typeof queryDescriptorSchema>;

/**
 * Parse and validate a raw request body into a QueryDescriptor.
 * Throws a formatted error string on validation failure.
 */
export function parseAndValidateQuery(body: unknown): {
  table: string;
  filters: { column: string; op: "eq" | "neq" | "gt" | "gte" | "lt" | "lte" | "in"; value: number | string | (number | string)[] }[];
  projections: string[];
  sortColumn?: string;
  sortDirection?: "asc" | "desc";
  limit?: number;
  offset?: number;
  vectorSearch?: { column: string; queryVector: number[] | Float32Array; topK: number };
  aggregates?: { fn: "sum" | "avg" | "min" | "max" | "count"; column: string; alias?: string }[];
  groupBy?: string[];
  cacheTTL?: number;
} {
  const result = queryDescriptorSchema.safeParse(body);
  if (!result.success) {
    const issues = result.error.issues.map(i =>
      `${i.path.join(".")}: ${i.message}`
    ).join("; ");
    throw new Error(`Invalid query: ${issues}`);
  }

  const data = result.data;
  // Merge `select` alias into `projections`
  const projections = data.projections.length > 0
    ? data.projections
    : (data.select ?? []);

  return {
    table: data.table,
    filters: data.filters,
    projections,
    sortColumn: data.sortColumn,
    sortDirection: data.sortDirection,
    limit: data.limit,
    offset: data.offset,
    vectorSearch: data.vectorSearch,
    aggregates: data.aggregates,
    groupBy: data.groupBy,
    cacheTTL: data.cacheTTL,
  };
}
