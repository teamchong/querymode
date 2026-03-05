import type { Row } from "./types.js";
import type { QueryDescriptor } from "./client.js";

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
  /** For stddev/variance: sum of squared values (Welford's online algorithm) */
  sumSq?: number;
  /** For median/percentile: collected values (exact mode) */
  values?: number[];
  /** For count_distinct: set of seen values */
  distinctSet?: Set<string>;
  /** For percentile: target percentile (0-1) */
  percentileTarget?: number;
}

export function initPartialAggState(
  fn: PartialAggState["fn"],
  column: string,
  opts?: { percentileTarget?: number },
): PartialAggState {
  const state: PartialAggState = { fn, column, sum: 0, count: 0, min: Infinity, max: -Infinity };
  if (fn === "stddev" || fn === "variance") state.sumSq = 0;
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
  state.sum += value;
  state.count++;
  if (value < state.min) state.min = value;
  if (value > state.max) state.max = value;
  if (state.sumSq !== undefined) state.sumSq += value * value;
  if (state.values !== undefined) state.values.push(value);
  if (state.distinctSet !== undefined) {
    state.distinctSet.add(rawValue !== undefined ? String(rawValue) : String(value));
  }
}

function resolveValue(state: PartialAggState): number {
  switch (state.fn) {
    case "sum":
      return state.sum;
    case "avg":
      return state.count === 0 ? 0 : state.sum / state.count;
    case "min":
      return state.min;
    case "max":
      return state.max;
    case "count":
      return state.count;
    case "count_distinct":
      return state.distinctSet?.size ?? 0;
    case "stddev": {
      if (state.count < 2) return 0;
      const mean = state.sum / state.count;
      const variance = (state.sumSq ?? 0) / state.count - mean * mean;
      return Math.sqrt(Math.max(0, variance));
    }
    case "variance": {
      if (state.count < 2) return 0;
      const mean = state.sum / state.count;
      return (state.sumSq ?? 0) / state.count - mean * mean;
    }
    case "median": {
      const vals = state.values ?? [];
      if (vals.length === 0) return 0;
      vals.sort((a, b) => a - b);
      const mid = vals.length >> 1;
      return vals.length % 2 === 0 ? (vals[mid - 1] + vals[mid]) / 2 : vals[mid];
    }
    case "percentile": {
      const vals = state.values ?? [];
      if (vals.length === 0) return 0;
      vals.sort((a, b) => a - b);
      const p = state.percentileTarget ?? 0.5;
      const idx = p * (vals.length - 1);
      const lo = Math.floor(idx);
      const hi = Math.ceil(idx);
      if (lo === hi) return vals[lo];
      return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo);
    }
  }
}

function aliasFor(agg: { fn: string; column: string; alias?: string }): string {
  return agg.alias ?? `${agg.fn}_${agg.column}`;
}

export function computePartialAgg(
  rows: Row[],
  query: QueryDescriptor,
): PartialAgg {
  const aggregates = query.aggregates ?? [];

  if (!query.groupBy || query.groupBy.length === 0) {
    const states = aggregates.map((agg) =>
      initPartialAggState(agg.fn, agg.column, { percentileTarget: agg.percentileTarget }),
    );
    for (const row of rows) {
      for (let i = 0; i < aggregates.length; i++) {
        const col = aggregates[i].column;
        if (col === "*") {
          // count(*) — always count the row
          states[i].count++;
        } else {
          const val = row[col];
          if (typeof val === "number") {
            updatePartialAgg(states[i], val, val);
          } else if (typeof val === "bigint") {
            updatePartialAgg(states[i], Number(val), val);
          } else if (aggregates[i].fn === "count_distinct" && val !== null && val !== undefined) {
            // count_distinct handles non-numeric values too
            updatePartialAgg(states[i], 0, val);
          }
        }
      }
    }
    return { states };
  }

  const groups = new Map<string, PartialAggState[]>();
  const groupCols = query.groupBy;

  for (const row of rows) {
    const key = groupCols.map((c) => String(row[c] ?? "")).join("\x00");
    let states = groups.get(key);
    if (!states) {
      states = aggregates.map((agg) =>
        initPartialAggState(agg.fn, agg.column, { percentileTarget: agg.percentileTarget }),
      );
      groups.set(key, states);
    }
    for (let i = 0; i < aggregates.length; i++) {
      const col = aggregates[i].column;
      if (col === "*") {
        states[i].count++;
      } else {
        const val = row[col];
        if (typeof val === "number") {
          updatePartialAgg(states[i], val, val);
        } else if (typeof val === "bigint") {
          updatePartialAgg(states[i], Number(val), val);
        } else if (aggregates[i].fn === "count_distinct" && val !== null && val !== undefined) {
          updatePartialAgg(states[i], 0, val);
        }
      }
    }
  }

  return { states: [], groups };
}

function mergeStates(
  target: PartialAggState[],
  source: PartialAggState[],
): void {
  for (let i = 0; i < target.length; i++) {
    target[i].sum += source[i].sum;
    target[i].count += source[i].count;
    if (source[i].min < target[i].min) target[i].min = source[i].min;
    if (source[i].max > target[i].max) target[i].max = source[i].max;
    // Extended aggregate state merge
    if (target[i].sumSq !== undefined && source[i].sumSq !== undefined) {
      target[i].sumSq! += source[i].sumSq!;
    }
    if (target[i].values !== undefined && source[i].values !== undefined) {
      target[i].values!.push(...source[i].values!);
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
      row[groupCols[i]] = keyParts[i];
    }
    for (let i = 0; i < states.length; i++) {
      const alias = aliasFor(aggregates[i]);
      row[alias] = resolveValue(states[i]);
    }
    rows.push(row);
  }

  return rows;
}
