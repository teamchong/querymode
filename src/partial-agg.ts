import type { Row } from "./types.js";
import { groupKey, NULL_SENTINEL } from "./types.js";
import type { QueryDescriptor } from "./client.js";
import type { ColumnarBatch, DecodedValue } from "./operators.js";
import { identityIndices } from "./operators.js";

export interface PartialAgg {
  states: PartialAggState[];
  groups?: Map<string, PartialAggState[]>;
}

export interface PartialAggState {
  fn: "sum" | "avg" | "min" | "max" | "count" | "count_distinct" | "stddev" | "variance" | "median" | "percentile";
  column: string;
  sum: number;
  count: number;
  min: number;
  max: number;
  /** For stddev/variance: Welford's M2 = sum of squared deviations from mean */
  m2?: number;
  /** For median/percentile: collected values (exact mode) */
  values?: number[];
  /** For count_distinct: set of seen values */
  distinctSet?: Set<string>;
  /** For percentile: target percentile (0-1) */
  percentileTarget?: number;
  /** For min/max on string columns */
  strMin?: string;
  strMax?: string;
}

export function initPartialAggState(
  fn: PartialAggState["fn"],
  column: string,
  opts?: { percentileTarget?: number },
): PartialAggState {
  const state: PartialAggState = { fn, column, sum: 0, count: 0, min: Infinity, max: -Infinity };
  if (fn === "stddev" || fn === "variance") state.m2 = 0;
  if (fn === "median" || fn === "percentile") {
    state.values = [];
    if (fn === "percentile" && opts?.percentileTarget !== undefined) {
      state.percentileTarget = opts.percentileTarget;
    }
  }
  if (fn === "count_distinct") state.distinctSet = new Set();
  return state;
}

export function updatePartialAgg(
  state: PartialAggState,
  value: number,
  rawValue?: unknown,
): void {
  if (state.m2 !== undefined) {
    // Welford's online algorithm — compute before updating sum/count
    const oldMean = state.count > 0 ? state.sum / state.count : 0;
    const delta = value - oldMean;
    state.sum += value;
    state.count++;
    const newMean = state.sum / state.count;
    state.m2 += delta * (value - newMean);
  } else {
    state.sum += value;
    state.count++;
  }
  if (value < state.min) state.min = value;
  if (value > state.max) state.max = value;
  if (state.values !== undefined) state.values.push(value);
  if (state.distinctSet !== undefined) {
    state.distinctSet.add(rawValue !== undefined ? String(rawValue) : String(value));
  }
}

function resolveValue(state: PartialAggState): number | string | null {
  // count/count_distinct always return a number (0 for empty)
  if (state.fn === "count") return state.count;
  if (state.fn === "count_distinct") return state.distinctSet?.size ?? 0;
  // All other aggregates return null when no non-null values were seen (SQL standard)
  if (state.count === 0) return null;
  switch (state.fn) {
    case "sum": return state.sum;
    case "avg": return state.sum / state.count;
    case "min": return state.strMin !== undefined ? state.strMin : state.min;
    case "max": return state.strMax !== undefined ? state.strMax : state.max;
    case "stddev": return state.count < 2 ? 0 : Math.sqrt(Math.max(0, (state.m2 ?? 0) / state.count));
    case "variance": return state.count < 2 ? 0 : (state.m2 ?? 0) / state.count;
    case "median": {
      const vals = state.values ?? [];
      vals.sort((a, b) => a - b);
      const mid = vals.length >> 1;
      return vals.length % 2 === 0 ? (vals[mid - 1] + vals[mid]) / 2 : vals[mid];
    }
    case "percentile": {
      const vals = state.values ?? [];
      vals.sort((a, b) => a - b);
      const p = state.percentileTarget ?? 0.5;
      const idx = p * (vals.length - 1);
      const lo = Math.floor(idx), hi = Math.ceil(idx);
      return lo === hi ? vals[lo] : vals[lo] + (vals[hi] - vals[lo]) * (idx - lo);
    }
  }
}

function aliasFor(agg: { fn: string; column: string; alias?: string }): string {
  return agg.alias ?? `${agg.fn}_${agg.column}`;
}

/** Feed a single value into agg states for one row. */
function feedAggStates(
  states: PartialAggState[],
  aggregates: { fn: PartialAggState["fn"]; column: string }[],
  getVal: (col: string, aggIdx: number) => unknown,
): void {
  for (let i = 0; i < aggregates.length; i++) {
    const col = aggregates[i].column;
    if (col === "*") { states[i].count++; continue; }
    const val = getVal(col, i);
    if (typeof val === "number") {
      updatePartialAgg(states[i], val, val);
    } else if (typeof val === "bigint") {
      updatePartialAgg(states[i], Number(val), val);
    } else if (val !== null && val !== undefined) {
      if (aggregates[i].fn === "count_distinct") {
        updatePartialAgg(states[i], 0, val);
      } else if (aggregates[i].fn === "count") {
        states[i].count++;
      } else if (typeof val === "string" && (aggregates[i].fn === "min" || aggregates[i].fn === "max")) {
        states[i].count++;
        if (states[i].strMin === undefined || val < states[i].strMin!) states[i].strMin = val;
        if (states[i].strMax === undefined || val > states[i].strMax!) states[i].strMax = val;
      }
    }
  }
}

function initStates(aggregates: { fn: PartialAggState["fn"]; column: string; percentileTarget?: number }[]): PartialAggState[] {
  return aggregates.map((agg) =>
    initPartialAggState(agg.fn, agg.column, { percentileTarget: agg.percentileTarget }),
  );
}

export function computePartialAgg(
  rows: Row[],
  query: QueryDescriptor,
): PartialAgg {
  const aggregates = query.aggregates ?? [];

  if (!query.groupBy || query.groupBy.length === 0) {
    const states = initStates(aggregates);
    for (const row of rows) feedAggStates(states, aggregates, col => row[col]);
    return { states };
  }

  const groups = new Map<string, PartialAggState[]>();
  const groupCols = query.groupBy;
  for (const row of rows) {
    const key = groupKey(row, groupCols);
    let states = groups.get(key);
    if (!states) { states = initStates(aggregates); groups.set(key, states); }
    feedAggStates(states, aggregates, col => row[col]);
  }
  return { states: [], groups };
}

export function computePartialAggColumnar(
  batch: ColumnarBatch,
  query: QueryDescriptor,
): PartialAgg {
  const aggregates = query.aggregates ?? [];
  const indices = batch.selection ?? identityIndices(batch.rowCount);
  const aggArrays = aggregates.map(a => a.column === "*" ? null : (batch.columns.get(a.column) ?? null));

  if (!query.groupBy || query.groupBy.length === 0) {
    const states = initStates(aggregates);
    for (const idx of indices) feedAggStates(states, aggregates, (_, i) => aggArrays[i] ? aggArrays[i]![idx] : undefined);
    return { states };
  }

  const groups = new Map<string, PartialAggState[]>();
  const groupCols = query.groupBy;
  const groupArrays = groupCols.map(c => batch.columns.get(c) ?? null);

  for (const idx of indices) {
    let key = "";
    for (let g = 0; g < groupCols.length; g++) {
      if (g > 0) key += "\x00";
      const v = groupArrays[g] ? groupArrays[g]![idx] : null;
      key += v === null || v === undefined ? NULL_SENTINEL : String(v);
    }
    let states = groups.get(key);
    if (!states) { states = initStates(aggregates); groups.set(key, states); }
    feedAggStates(states, aggregates, (_, i) => aggArrays[i] ? aggArrays[i]![idx] : undefined);
  }
  return { states: [], groups };
}

function mergeStates(
  target: PartialAggState[],
  source: PartialAggState[],
): void {
  for (let i = 0; i < target.length; i++) {
    // Parallel Welford merge for m2 (must be computed before sum/count merge)
    if (target[i].m2 !== undefined && source[i].m2 !== undefined) {
      const nA = target[i].count, nB = source[i].count;
      if (nA > 0 && nB > 0) {
        const delta = (source[i].sum / nB) - (target[i].sum / nA);
        target[i].m2 = target[i].m2! + source[i].m2! + delta * delta * nA * nB / (nA + nB);
      } else {
        target[i].m2! += source[i].m2!;
      }
    }
    target[i].sum += source[i].sum;
    target[i].count += source[i].count;
    if (source[i].min < target[i].min) target[i].min = source[i].min;
    if (source[i].max > target[i].max) target[i].max = source[i].max;
    if (source[i].strMin !== undefined) {
      if (target[i].strMin === undefined || source[i].strMin! < target[i].strMin!) target[i].strMin = source[i].strMin;
    }
    if (source[i].strMax !== undefined) {
      if (target[i].strMax === undefined || source[i].strMax! > target[i].strMax!) target[i].strMax = source[i].strMax;
    }
    if (target[i].values !== undefined && source[i].values !== undefined) {
      const src = source[i].values!;
      const tgt = target[i].values!;
      for (let j = 0; j < src.length; j++) tgt.push(src[j]);
    }
    if (target[i].distinctSet !== undefined && source[i].distinctSet !== undefined) {
      for (const v of source[i].distinctSet!) target[i].distinctSet!.add(v);
    }
  }
}

export function mergePartialAggs(partials: PartialAgg[]): PartialAgg {
  if (partials.length === 0) return { states: [] };

  const hasGroups = partials.some((p) => p.groups && p.groups.size > 0);

  if (!hasGroups) {
    const merged = partials[0].states.map((s) => ({ ...s }));
    for (let i = 1; i < partials.length; i++) {
      mergeStates(merged, partials[i].states);
    }
    return { states: merged };
  }

  const mergedGroups = new Map<string, PartialAggState[]>();
  for (const partial of partials) {
    if (!partial.groups) continue;
    for (const [key, states] of partial.groups) {
      const existing = mergedGroups.get(key);
      if (!existing) {
        mergedGroups.set(key, states.map((s) => ({ ...s })));
      } else {
        mergeStates(existing, states);
      }
    }
  }

  return { states: [], groups: mergedGroups };
}

export function finalizePartialAgg(
  agg: PartialAgg,
  query: QueryDescriptor,
): Row[] {
  const aggregates = query.aggregates ?? [];

  if (!agg.groups || agg.groups.size === 0) {
    const row: Row = {};
    for (let i = 0; i < agg.states.length; i++) {
      const alias = aliasFor(aggregates[i]);
      row[alias] = resolveValue(agg.states[i]);
    }
    return [row];
  }

  const groupCols = query.groupBy ?? [];
  const rows: Row[] = [];

  for (const [key, states] of agg.groups) {
    const row: Row = {};
    const keyParts = key.split("\x00");
    for (let i = 0; i < groupCols.length; i++) {
      const part = keyParts[i];
      if (part === NULL_SENTINEL) {
        row[groupCols[i]] = null;
      } else {
        // Attempt to restore numeric types
        const num = Number(part);
        row[groupCols[i]] = part !== "" && !isNaN(num) && String(num) === part ? num : part;
      }
    }
    for (let i = 0; i < states.length; i++) {
      const alias = aliasFor(aggregates[i]);
      row[alias] = resolveValue(states[i]);
    }
    rows.push(row);
  }

  return rows;
}
