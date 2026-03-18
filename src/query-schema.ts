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
  op: z.enum(["eq", "neq", "gt", "gte", "lt", "lte", "in", "not_in", "between", "not_between", "like", "not_like", "is_null", "is_not_null"]),
  value: z.union([
    z.number(),
    z.string(),
    z.array(z.union([z.number(), z.string()])),
  ]).optional(), // optional for unary ops (is_null, is_not_null)
});

const aggregateOpSchema = z.object({
  fn: z.enum(["sum", "avg", "min", "max", "count", "count_distinct", "stddev", "variance", "median", "percentile"]),
  column: z.string().min(1, "Aggregate column name cannot be empty"),
  alias: z.string().optional(),
  percentileTarget: z.number().min(0).max(1).optional(),
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
  table: z.string().min(1, "Table name is required")
    .refine(s => !s.includes("..") && !s.startsWith("/"), "Invalid table name"),
  filters: z.array(filterOpSchema).default([]),
  projections: z.array(z.string()).default([]),
  select: z.array(z.string()).optional(), // alias for projections
  sortColumn: z.string().optional(),
  sortDirection: z.enum(["asc", "desc"]).optional(),
  orderBy: z.object({ column: z.string(), desc: z.boolean().optional() }).optional(), // alias for sortColumn/sortDirection
  limit: z.number().int().nonnegative().optional(),
  offset: z.number().int().nonnegative().optional(),
  vectorSearch: vectorSearchSchema.optional(),
  aggregates: z.array(aggregateOpSchema).optional(),
  filterGroups: z.array(z.array(filterOpSchema)).optional(),
  distinct: z.array(z.string()).optional(),
  groupBy: z.array(z.string()).optional(),
  cacheTTL: z.number().int().positive().optional(),
});

/**
 * Parse and validate a raw request body into a QueryDescriptor.
 * Throws a formatted error string on validation failure.
 */
export function parseAndValidateQuery(body: unknown): {
  table: string;
  filters: { column: string; op: "eq" | "neq" | "gt" | "gte" | "lt" | "lte" | "in" | "not_in" | "between" | "not_between" | "like" | "not_like" | "is_null" | "is_not_null"; value?: number | string | (number | string)[] }[];
  projections: string[];
  sortColumn?: string;
  sortDirection?: "asc" | "desc";
  limit?: number;
  offset?: number;
  vectorSearch?: { column: string; queryVector: number[] | Float32Array; topK: number };
  aggregates?: { fn: "sum" | "avg" | "min" | "max" | "count" | "count_distinct" | "stddev" | "variance" | "median" | "percentile"; column: string; alias?: string; percentileTarget?: number }[];
  filterGroups?: { column: string; op: string; value?: number | string | (number | string)[] }[][];
  distinct?: string[];
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

  // Merge `orderBy` alias into `sortColumn` / `sortDirection`
  const sortColumn = data.sortColumn ?? data.orderBy?.column;
  const sortDirection = data.sortDirection ?? (data.orderBy?.desc ? "desc" : data.orderBy ? "asc" : undefined);

  return {
    table: data.table,
    filters: data.filters,
    projections,
    sortColumn,
    sortDirection,
    limit: data.limit,
    offset: data.offset,
    vectorSearch: data.vectorSearch,
    aggregates: data.aggregates,
    distinct: data.distinct,
    groupBy: data.groupBy,
    filterGroups: data.filterGroups,
    cacheTTL: data.cacheTTL,
  };
}
