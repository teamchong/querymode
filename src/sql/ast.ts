/** SQL AST types — ported from lanceql/src/sql/ast.zig (SELECT subset only) */

export type SqlValue =
  | { type: "null" }
  | { type: "integer"; value: number }
  | { type: "float"; value: number }
  | { type: "string"; value: string }
  | { type: "boolean"; value: boolean };

export type BinaryOp =
  | "add" | "subtract" | "multiply" | "divide" | "concat"
  | "eq" | "ne" | "lt" | "le" | "gt" | "ge"
  | "and" | "or"
  | "like";

export type UnaryOp = "not" | "minus" | "is_null" | "is_not_null";

export type SqlExpr =
  | { kind: "value"; value: SqlValue }
  | { kind: "column"; table?: string; name: string }
  | { kind: "binary"; op: BinaryOp; left: SqlExpr; right: SqlExpr }
  | { kind: "unary"; op: UnaryOp; operand: SqlExpr }
  | { kind: "call"; name: string; args: SqlExpr[]; distinct?: boolean; window?: SqlWindowSpec }
  | { kind: "in_list"; expr: SqlExpr; values: SqlExpr[]; negated: boolean }
  | { kind: "between"; expr: SqlExpr; low: SqlExpr; high: SqlExpr }
  | { kind: "case_expr"; operand?: SqlExpr; whenClauses: CaseWhen[]; elseResult?: SqlExpr }
  | { kind: "cast"; expr: SqlExpr; targetType: string }
  | { kind: "star" }
  | { kind: "near"; column: SqlExpr; vector: number[]; topK?: number }
  | { kind: "exists"; subquery: SelectStmt; negated: boolean }
  | { kind: "parameter"; index: number };

export interface CaseWhen {
  condition: SqlExpr;
  result: SqlExpr;
}

export type FrameBound =
  | { type: "unbounded_preceding" }
  | { type: "current_row" }
  | { type: "unbounded_following" }
  | { type: "preceding"; offset: number }
  | { type: "following"; offset: number };

export interface SqlWindowSpec {
  partitionBy?: string[];
  orderBy?: SqlOrderBy[];
  frame?: { type: "rows" | "range"; start: FrameBound; end?: FrameBound };
}

export interface SelectItem {
  expr: SqlExpr;
  alias?: string;
}

export type JoinType = "inner" | "left" | "right" | "full" | "cross";

export interface JoinClause {
  joinType: JoinType;
  table: TableRef;
  onCondition?: SqlExpr;
  natural?: boolean;
  using?: string[];
}

export type TableRef =
  | { kind: "simple"; name: string; alias?: string }
  | { kind: "join"; left: TableRef; join: JoinClause };

export interface SqlOrderBy {
  column: string;
  direction: "asc" | "desc";
}

export interface SqlGroupBy {
  columns: string[];
  having?: SqlExpr;
}

export type SetOperationType = "union_all" | "union_distinct" | "intersect" | "except";

export interface SetOperation {
  opType: SetOperationType;
  right: SelectStmt;
}

export interface CteDef {
  name: string;
  query: SelectStmt;
}

export interface SelectStmt {
  distinct: boolean;
  columns: SelectItem[];
  from: TableRef;
  where?: SqlExpr;
  groupBy?: SqlGroupBy;
  orderBy?: SqlOrderBy[];
  limit?: number;
  offset?: number;
  setOperation?: SetOperation;
  ctes?: CteDef[];
}

export interface ShowVersionsStmt {
  table: string;
  limit?: number;
}

export interface DiffStmt {
  table: string;
  fromVersion: number;
  toVersion?: number;
  limit?: number;
}

export type SqlStatement =
  | { kind: "select"; stmt: SelectStmt }
  | { kind: "show_versions"; stmt: ShowVersionsStmt }
  | { kind: "diff"; stmt: DiffStmt };

/**
 * Deterministic string key for an aggregate argument expression.
 * Returns "col" for column refs, "*" for star, or a serialized form
 * for complex expressions (e.g. "add(a,b)" for a + b).
 * Used by both compileAggregate and rewriteAggregatesAsColumns so
 * their output column names always match.
 */
export function aggArgKey(expr: SqlExpr): string {
  switch (expr.kind) {
    case "star": return "*";
    case "column": return expr.name;
    case "value": {
      const v = expr.value;
      return v.type === "null" ? "null" : String(v.value);
    }
    case "binary": return `${expr.op}(${aggArgKey(expr.left)},${aggArgKey(expr.right)})`;
    case "unary": return `${expr.op}(${aggArgKey(expr.operand)})`;
    case "call": return `${expr.name}(${expr.args.map(aggArgKey).join(",")})`;
    case "cast": return `cast(${aggArgKey(expr.expr)},${expr.targetType})`;
    default: return "_expr";
  }
}
