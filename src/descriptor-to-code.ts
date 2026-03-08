import type { QueryDescriptor } from "./client.js";
import type { FilterOp, AggregateOp, WindowSpec } from "./types.js";

/**
 * Convert a QueryDescriptor back to fluent DataFrame builder code.
 *
 * Produces compact, readable TypeScript that recreates the exact same query.
 * Use cases: logging, debugging, context compression for LLM agents,
 * auto-generating documentation examples from real queries.
 */
export function descriptorToCode(
  desc: QueryDescriptor,
  opts?: { variableName?: string; tableFn?: string },
): string {
  const varName = opts?.variableName ?? "result";
  const tableFn = opts?.tableFn ?? "qm";
  const chains: string[] = [];

  // Table
  chains.push(`${tableFn}\n  .table(${str(desc.table)})`);

  // Version
  if (desc.version !== undefined) {
    chains.push(`.version(${desc.version})`);
  }

  // Cache
  if (desc.cacheTTL !== undefined) {
    chains.push(`.cache({ ttl: ${desc.cacheTTL} })`);
  }

  // Filters
  for (const f of desc.filters) {
    chains.push(filterToChain(f));
  }

  // Filter groups (OR logic)
  if (desc.filterGroups && desc.filterGroups.length > 0) {
    const groupArgs = desc.filterGroups.map(group =>
      `[${group.map(f => filterOpLiteral(f)).join(", ")}]`
    ).join(", ");
    chains.push(`.whereOr(${groupArgs})`);
  }

  // Computed columns — callbacks cannot be serialized, so we emit
  // a .computed() call with the alias and an identity function.
  // The caller must replace the function body with the real logic.
  if (desc.computedColumns && desc.computedColumns.length > 0) {
    for (const cc of desc.computedColumns) {
      chains.push(`.computed(${str(cc.alias)}, (row) => row[${str(cc.alias)}])`);
    }
  }

  // Join
  if (desc.join) {
    const rightCode = descriptorToCode(
      desc.join.right,
      { variableName: "_right", tableFn },
    );
    const rightChain = extractChainBody(rightCode);
    const joinType = desc.join.type && desc.join.type !== "inner" ? `, ${str(desc.join.type)}` : "";
    chains.push(`.join(\n    ${rightChain},\n    { left: ${str(desc.join.leftKey)}, right: ${str(desc.join.rightKey)} }${joinType},\n  )`);
  }

  // SubqueryIn
  if (desc.subqueryIn && desc.subqueryIn.length > 0) {
    for (const sq of desc.subqueryIn) {
      const values = [...sq.valueSet].map(v => str(v)).join(", ");
      chains.push(`.filterIn(${str(sq.column)}, ${tableFn}.table("_subquery").whereIn(${str(sq.column)}, [${values}]))`);
    }
  }

  // Window functions
  if (desc.windows && desc.windows.length > 0) {
    for (const w of desc.windows) {
      chains.push(`.window(${windowSpecLiteral(w)})`);
    }
  }

  // Distinct
  if (desc.distinct) {
    if (desc.distinct.length > 0) {
      chains.push(`.distinct(${desc.distinct.map(str).join(", ")})`);
    } else {
      chains.push(`.distinct()`);
    }
  }

  // Group by
  if (desc.groupBy && desc.groupBy.length > 0) {
    chains.push(`.groupBy(${desc.groupBy.map(str).join(", ")})`);
  }

  // Aggregates
  if (desc.aggregates && desc.aggregates.length > 0) {
    for (const agg of desc.aggregates) {
      chains.push(aggToChain(agg));
    }
  }

  // Set operation
  if (desc.setOperation) {
    const rightCode = descriptorToCode(
      desc.setOperation.right,
      { variableName: "_right", tableFn },
    );
    const rightChain = extractChainBody(rightCode);
    switch (desc.setOperation.mode) {
      case "union":
        chains.push(`.union(\n    ${rightChain},\n  )`);
        break;
      case "union_all":
        chains.push(`.union(\n    ${rightChain},\n    true,\n  )`);
        break;
      case "intersect":
        chains.push(`.intersect(\n    ${rightChain},\n  )`);
        break;
      case "except":
        chains.push(`.except(\n    ${rightChain},\n  )`);
        break;
    }
  }

  // Projections
  if (desc.projections.length > 0) {
    chains.push(`.select(${desc.projections.map(str).join(", ")})`);
  }

  // Sort
  if (desc.sortColumn) {
    const dir = desc.sortDirection === "desc" ? `, "desc"` : "";
    chains.push(`.sort(${str(desc.sortColumn)}${dir})`);
  }

  // Offset
  if (desc.offset) {
    chains.push(`.offset(${desc.offset})`);
  }

  // Limit
  if (desc.limit !== undefined) {
    chains.push(`.limit(${desc.limit})`);
  }

  // Vector search
  if (desc.vectorSearch) {
    const vs = desc.vectorSearch;
    const vecStr = `new Float32Array([${[...vs.queryVector].join(", ")}])`;
    const optsparts: string[] = [];
    if (vs.metric && vs.metric !== "cosine") optsparts.push(`metric: ${str(vs.metric)}`);
    if (vs.nprobe) optsparts.push(`nprobe: ${vs.nprobe}`);
    if (vs.efSearch) optsparts.push(`efSearch: ${vs.efSearch}`);
    const optsStr = optsparts.length > 0 ? `, { ${optsparts.join(", ")} }` : "";
    chains.push(`.vector(${str(vs.column)}, ${vecStr}, ${vs.topK}${optsStr})`);
  }

  // Pipe stages — callbacks cannot be serialized, so we emit
  // a .pipe() call with an identity function.
  // The caller must replace the function body with the real logic.
  if (desc.pipeStages && desc.pipeStages.length > 0) {
    for (let i = 0; i < desc.pipeStages.length; i++) {
      chains.push(`.pipe((upstream) => upstream)`);
    }
  }

  // Terminal
  chains.push(`.collect()`);

  const chainStr = chains.join("\n  ");
  return `const ${varName} = await ${chainStr}`;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Extract the fluent chain body from generated code (strips variable declaration and terminal). */
function extractChainBody(code: string): string {
  // Strip "const <var> = await " prefix and trailing ".collect()"
  const awaitIdx = code.indexOf("await ");
  const body = awaitIdx >= 0 ? code.slice(awaitIdx + 6) : code;
  const lines = body.split("\n").map(l => l.trim()).filter(Boolean);
  // Remove trailing .collect()
  if (lines.length > 0 && lines[lines.length - 1] === ".collect()") {
    lines.pop();
  }
  return lines.join("\n    ");
}

function str(v: unknown): string {
  if (typeof v === "bigint") return `${v}n`;
  return JSON.stringify(v);
}

function filterValueStr(value: FilterOp["value"]): string {
  if (Array.isArray(value)) {
    return `[${value.map(v => typeof v === "bigint" ? `${v}n` : JSON.stringify(v)).join(", ")}]`;
  }
  return str(value);
}

function filterToChain(f: FilterOp): string {
  switch (f.op) {
    case "is_null":
      return `.whereNull(${str(f.column)})`;
    case "is_not_null":
      return `.whereNotNull(${str(f.column)})`;
    case "in":
      return `.whereIn(${str(f.column)}, ${filterValueStr(f.value)})`;
    case "not_in":
      return `.whereNotIn(${str(f.column)}, ${filterValueStr(f.value)})`;
    case "between":
      if (Array.isArray(f.value) && f.value.length === 2) {
        return `.whereBetween(${str(f.column)}, ${str(f.value[0])}, ${str(f.value[1])})`;
      }
      return `.filter(${str(f.column)}, "between", ${filterValueStr(f.value)})`;
    case "not_between":
      if (Array.isArray(f.value) && f.value.length === 2) {
        return `.whereNotBetween(${str(f.column)}, ${str(f.value[0])}, ${str(f.value[1])})`;
      }
      return `.filter(${str(f.column)}, "not_between", ${filterValueStr(f.value)})`;
    case "like":
      return `.whereLike(${str(f.column)}, ${str(f.value)})`;
    case "not_like":
      return `.whereNotLike(${str(f.column)}, ${str(f.value)})`;
    default:
      return `.filter(${str(f.column)}, ${str(f.op)}, ${filterValueStr(f.value)})`;
  }
}

function filterOpLiteral(f: FilterOp): string {
  return `{ column: ${str(f.column)}, op: ${str(f.op)}, value: ${filterValueStr(f.value)} }`;
}

function aggToChain(agg: AggregateOp): string {
  if (agg.fn === "percentile") {
    const alias = agg.alias ? `, ${str(agg.alias)}` : "";
    return `.percentile(${str(agg.column)}, ${agg.percentileTarget ?? 0.5}${alias})`;
  }
  const alias = agg.alias ? `, ${str(agg.alias)}` : "";
  return `.aggregate(${str(agg.fn)}, ${str(agg.column)}${alias})`;
}

function windowSpecLiteral(w: WindowSpec): string {
  const parts: string[] = [];
  parts.push(`fn: ${str(w.fn)}`);
  if (w.column) parts.push(`column: ${str(w.column)}`);
  if (w.partitionBy.length > 0) {
    parts.push(`partitionBy: [${w.partitionBy.map(str).join(", ")}]`);
  } else {
    parts.push(`partitionBy: []`);
  }
  if (w.orderBy.length > 0) {
    const obs = w.orderBy.map(o =>
      `{ column: ${str(o.column)}, direction: ${str(o.direction)} }`
    ).join(", ");
    parts.push(`orderBy: [${obs}]`);
  } else {
    parts.push(`orderBy: []`);
  }
  parts.push(`alias: ${str(w.alias)}`);
  if (w.args) {
    const argParts: string[] = [];
    if (w.args.offset !== undefined) argParts.push(`offset: ${w.args.offset}`);
    if (w.args.default_ !== undefined) argParts.push(`default_: ${str(w.args.default_)}`);
    if (argParts.length > 0) parts.push(`args: { ${argParts.join(", ")} }`);
  }
  if (w.frame) {
    const startStr = typeof w.frame.start === "number" ? String(w.frame.start) : str(w.frame.start);
    const endStr = typeof w.frame.end === "number" ? String(w.frame.end) : str(w.frame.end);
    parts.push(`frame: { type: ${str(w.frame.type)}, start: ${startStr}, end: ${endStr} }`);
  }
  return `{\n    ${parts.join(",\n    ")},\n  }`;
}
