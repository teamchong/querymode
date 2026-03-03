import type { Row } from "./types.js";
import type { QueryDescriptor } from "./client.js";

export interface PartialAgg {
  states: PartialAggState[];
  groups?: Map<string, PartialAggState[]>;
}

export interface PartialAggState {
  fn: "sum" | "avg" | "min" | "max" | "count";
  column: string;
  sum: number;
  count: number;
  min: number;
  max: number;
}

export function initPartialAggState(
  fn: PartialAggState["fn"],
  column: string,
): PartialAggState {
  return { fn, column, sum: 0, count: 0, min: Infinity, max: -Infinity };
}

export function updatePartialAgg(
  state: PartialAggState,
  value: number,
): void {
  state.sum += value;
  state.count++;
  if (value < state.min) state.min = value;
  if (value > state.max) state.max = value;
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
      initPartialAggState(agg.fn, agg.column),
    );
    for (const row of rows) {
      for (let i = 0; i < aggregates.length; i++) {
        const val = row[aggregates[i].column];
        if (typeof val === "number") {
          updatePartialAgg(states[i], val);
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
        initPartialAggState(agg.fn, agg.column),
      );
      groups.set(key, states);
    }
    for (let i = 0; i < aggregates.length; i++) {
      const val = row[aggregates[i].column];
      if (typeof val === "number") {
        updatePartialAgg(states[i], val);
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
