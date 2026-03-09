/** SQL Parser — recursive descent, ported from lanceql/src/sql/parser.zig */

import { tokenize, TokenType } from "./lexer.js";
import type { Token } from "./lexer.js";
import type {
  SelectStmt, SelectItem, SqlExpr, TableRef, JoinClause, JoinType,
  SqlOrderBy, SqlGroupBy, BinaryOp, UnaryOp, SetOperationType, SqlWindowSpec,
  SqlStatement, ShowVersionsStmt, DiffStmt,
} from "./ast.js";

export function parse(sql: string): SelectStmt {
  const tokens = tokenize(sql);
  const parser = new Parser(tokens, sql);
  const stmt = parser.parseSelect();
  // Allow optional trailing semicolon
  if (parser.check(TokenType.SEMICOLON)) parser.advance();
  if (!parser.check(TokenType.EOF)) {
    throw parser.error("Unexpected token after statement");
  }
  return stmt;
}

export function parseStatement(sql: string): SqlStatement {
  const tokens = tokenize(sql);
  const parser = new Parser(tokens, sql);
  const stmt = parser.parseStatementInternal();
  if (parser.check(TokenType.SEMICOLON)) parser.advance();
  if (!parser.check(TokenType.EOF)) {
    throw parser.error("Unexpected token after statement");
  }
  return stmt;
}

class Parser {
  private tokens: Token[];
  private pos = 0;
  private sql: string;
  private paramCount = 0;

  constructor(tokens: Token[], sql: string) {
    this.tokens = tokens;
    this.sql = sql;
  }

  // --- Token helpers ---

  private current(): Token {
    return this.tokens[this.pos] ?? this.tokens[this.tokens.length - 1];
  }

  advance(): void {
    if (this.pos < this.tokens.length - 1) this.pos++;
  }

  check(type: TokenType): boolean {
    return this.current().type === type;
  }

  private match(...types: TokenType[]): boolean {
    for (const t of types) {
      if (this.check(t)) {
        this.advance();
        return true;
      }
    }
    return false;
  }

  private expect(type: TokenType): Token {
    const tok = this.current();
    if (tok.type !== type) {
      throw this.error(`Expected ${TokenType[type]}, got '${tok.lexeme}'`);
    }
    this.advance();
    return tok;
  }

  error(message: string): SqlParseError {
    const tok = this.current();
    return new SqlParseError(`${message} at position ${tok.position}`, tok.position);
  }

  // --- Parsing ---

  parseStatementInternal(): SqlStatement {
    if (this.check(TokenType.SHOW)) {
      return { kind: "show_versions", stmt: this.parseShowVersions() };
    }
    if (this.check(TokenType.DIFF)) {
      return { kind: "diff", stmt: this.parseDiff() };
    }
    return { kind: "select", stmt: this.parseSelect() };
  }

  private parseShowVersions(): ShowVersionsStmt {
    this.expect(TokenType.SHOW);
    this.expect(TokenType.VERSIONS);
    this.expect(TokenType.FOR);
    const table = this.parseIdentifier();
    let limit: number | undefined;
    if (this.match(TokenType.LIMIT)) {
      const tok = this.expect(TokenType.NUMBER);
      limit = parseInt(tok.lexeme, 10);
    }
    return { table, limit };
  }

  private parseDiff(): DiffStmt {
    this.expect(TokenType.DIFF);
    const table = this.parseIdentifier();
    this.expect(TokenType.VERSION);
    const fromTok = this.expect(TokenType.NUMBER);
    const fromVersion = parseInt(fromTok.lexeme, 10);
    let toVersion: number | undefined;
    if (this.match(TokenType.AND)) {
      // Allow optional VERSION keyword after AND
      this.match(TokenType.VERSION);
      const toTok = this.expect(TokenType.NUMBER);
      toVersion = parseInt(toTok.lexeme, 10);
    }
    let limit: number | undefined;
    if (this.match(TokenType.LIMIT)) {
      const tok = this.expect(TokenType.NUMBER);
      limit = parseInt(tok.lexeme, 10);
    }
    return { table, fromVersion, toVersion, limit };
  }

  parseSelect(): SelectStmt {
    this.expect(TokenType.SELECT);

    const distinct = this.match(TokenType.DISTINCT);
    const columns = this.parseSelectList();
    this.expect(TokenType.FROM);
    const from = this.parseTableRef();

    let where: SqlExpr | undefined;
    if (this.match(TokenType.WHERE)) {
      where = this.parseExpr();
    }

    let groupBy: SqlGroupBy | undefined;
    if (this.check(TokenType.GROUP)) {
      groupBy = this.parseGroupBy();
    }

    let orderBy: SqlOrderBy[] | undefined;
    if (this.check(TokenType.ORDER)) {
      orderBy = this.parseOrderBy();
    }

    let limit: number | undefined;
    if (this.match(TokenType.LIMIT)) {
      const tok = this.expect(TokenType.NUMBER);
      limit = parseInt(tok.lexeme, 10);
    }

    let offset: number | undefined;
    if (this.match(TokenType.OFFSET)) {
      const tok = this.expect(TokenType.NUMBER);
      offset = parseInt(tok.lexeme, 10);
    }

    let setOperation: SelectStmt["setOperation"];
    if (this.check(TokenType.UNION) || this.check(TokenType.INTERSECT) || this.check(TokenType.EXCEPT)) {
      setOperation = this.parseSetOperation();
    }

    return { distinct, columns, from, where, groupBy, orderBy, limit, offset, setOperation };
  }

  private parseSelectList(): SelectItem[] {
    const items: SelectItem[] = [];
    do {
      items.push(this.parseSelectItem());
    } while (this.match(TokenType.COMMA));
    return items;
  }

  private parseSelectItem(): SelectItem {
    // Handle standalone * (SELECT *)
    if (this.check(TokenType.STAR)) {
      this.advance();
      return { expr: { kind: "star" } };
    }

    const expr = this.parseExpr();

    // Check for table.column after a column ref followed by DOT
    if (expr.kind === "column" && this.check(TokenType.DOT)) {
      this.advance();
      if (this.match(TokenType.STAR)) {
        // table.* — return a qualified star
        return { expr: { kind: "column", table: expr.name, name: "*" } };
      }
      // table.column
      const colName = this.parseIdentifier();
      const columnExpr: SqlExpr = { kind: "column", table: expr.name, name: colName };
      let alias: string | undefined;
      if (this.match(TokenType.AS)) {
        alias = this.parseIdentifier();
      } else if (this.check(TokenType.IDENTIFIER)) {
        alias = this.parseIdentifier();
      }
      return { expr: columnExpr, alias };
    }

    let alias: string | undefined;
    if (this.match(TokenType.AS)) {
      alias = this.parseIdentifier();
    } else if (
      this.check(TokenType.IDENTIFIER) &&
      !this.check(TokenType.FROM) &&
      !this.check(TokenType.COMMA)
    ) {
      // Implicit alias (no AS keyword)
      const next = this.current();
      if (next.type === TokenType.IDENTIFIER) {
        alias = this.parseIdentifier();
      }
    }
    return { expr, alias };
  }

  private parseIdentifier(): string {
    const tok = this.current();
    // Accept keywords that can be used as identifiers in alias position
    if (tok.type === TokenType.IDENTIFIER || tok.type === TokenType.STRING) {
      this.advance();
      // Strip quotes from identifiers
      if (tok.lexeme.startsWith('"') || tok.lexeme.startsWith("`")) {
        return tok.lexeme.slice(1, -1);
      }
      return tok.lexeme;
    }
    // Many keywords can be used as identifiers in certain positions
    if (isKeywordIdentifier(tok.type)) {
      this.advance();
      return tok.lexeme;
    }
    throw this.error(`Expected identifier, got '${tok.lexeme}'`);
  }

  // --- Table references ---

  private parseTableRef(): TableRef {
    let ref = this.parseSimpleTableRef();

    // Parse JOINs
    while (this.isJoinKeyword()) {
      const join = this.parseJoinClause(ref);
      ref = { kind: "join", left: ref, join };
    }

    return ref;
  }

  private parseSimpleTableRef(): TableRef {
    const name = this.parseIdentifier();
    let alias: string | undefined;
    if (this.match(TokenType.AS)) {
      alias = this.parseIdentifier();
    } else if (this.check(TokenType.IDENTIFIER) && !this.isJoinKeyword() && !this.isClauseKeyword()) {
      alias = this.parseIdentifier();
    }
    return { kind: "simple", name, alias };
  }

  private isJoinKeyword(): boolean {
    const t = this.current().type;
    return t === TokenType.JOIN || t === TokenType.LEFT || t === TokenType.RIGHT ||
      t === TokenType.INNER || t === TokenType.FULL || t === TokenType.CROSS ||
      t === TokenType.NATURAL;
  }

  private isClauseKeyword(): boolean {
    const t = this.current().type;
    return t === TokenType.WHERE || t === TokenType.GROUP || t === TokenType.ORDER ||
      t === TokenType.LIMIT || t === TokenType.OFFSET || t === TokenType.UNION ||
      t === TokenType.INTERSECT || t === TokenType.EXCEPT || t === TokenType.ON ||
      t === TokenType.HAVING || t === TokenType.USING;
  }

  private parseJoinClause(left: TableRef): JoinClause {
    let joinType: JoinType = "inner";
    let natural = false;

    if (this.match(TokenType.NATURAL)) {
      natural = true;
      // NATURAL can precede any join type or just JOIN
      if (this.match(TokenType.LEFT)) {
        this.match(TokenType.OUTER);
        joinType = "left";
      } else if (this.match(TokenType.RIGHT)) {
        this.match(TokenType.OUTER);
        joinType = "right";
      } else if (this.match(TokenType.FULL)) {
        this.match(TokenType.OUTER);
        joinType = "full";
      } else if (this.match(TokenType.INNER)) {
        joinType = "inner";
      }
      this.expect(TokenType.JOIN);
    } else if (this.match(TokenType.LEFT)) {
      this.match(TokenType.OUTER); // optional OUTER
      joinType = "left";
      this.expect(TokenType.JOIN);
    } else if (this.match(TokenType.RIGHT)) {
      this.match(TokenType.OUTER);
      joinType = "right";
      this.expect(TokenType.JOIN);
    } else if (this.match(TokenType.FULL)) {
      this.match(TokenType.OUTER);
      joinType = "full";
      this.expect(TokenType.JOIN);
    } else if (this.match(TokenType.CROSS)) {
      joinType = "cross";
      this.expect(TokenType.JOIN);
    } else if (this.match(TokenType.INNER)) {
      joinType = "inner";
      this.expect(TokenType.JOIN);
    } else {
      this.expect(TokenType.JOIN);
    }

    const table = this.parseSimpleTableRef();

    let onCondition: SqlExpr | undefined;
    let using: string[] | undefined;

    if (!natural && joinType !== "cross") {
      if (this.match(TokenType.ON)) {
        onCondition = this.parseExpr();
      } else if (this.match(TokenType.USING)) {
        this.expect(TokenType.LPAREN);
        using = [];
        do {
          using.push(this.parseIdentifier());
        } while (this.match(TokenType.COMMA));
        this.expect(TokenType.RPAREN);
      }
    }

    return { joinType, table, onCondition, natural: natural || undefined, using };
  }

  // --- GROUP BY ---

  private parseGroupBy(): SqlGroupBy {
    this.expect(TokenType.GROUP);
    this.expect(TokenType.BY);

    const columns: string[] = [];
    do {
      columns.push(this.parseIdentifier());
    } while (this.match(TokenType.COMMA));

    let having: SqlExpr | undefined;
    if (this.match(TokenType.HAVING)) {
      having = this.parseExpr();
    }

    return { columns, having };
  }

  // --- ORDER BY ---

  private parseOrderBy(): SqlOrderBy[] {
    this.expect(TokenType.ORDER);
    this.expect(TokenType.BY);

    const items: SqlOrderBy[] = [];
    do {
      const column = this.parseIdentifier();
      let direction: "asc" | "desc" = "asc";
      if (this.match(TokenType.ASC)) direction = "asc";
      else if (this.match(TokenType.DESC)) direction = "desc";
      items.push({ column, direction });
    } while (this.match(TokenType.COMMA));

    return items;
  }

  // --- Set operations ---

  private parseSetOperation(): SelectStmt["setOperation"] {
    let opType: SetOperationType;

    if (this.match(TokenType.UNION)) {
      opType = this.match(TokenType.ALL) ? "union_all" : "union_distinct";
    } else if (this.match(TokenType.INTERSECT)) {
      opType = "intersect";
    } else if (this.match(TokenType.EXCEPT)) {
      opType = "except";
    } else {
      throw this.error("Expected UNION, INTERSECT, or EXCEPT");
    }

    const right = this.parseSelect();
    return { opType, right };
  }

  // --- Expression parsing (precedence climbing) ---

  parseExpr(): SqlExpr {
    return this.parseOr();
  }

  private parseOr(): SqlExpr {
    let left = this.parseAnd();
    while (this.match(TokenType.OR)) {
      const right = this.parseAnd();
      left = { kind: "binary", op: "or", left, right };
    }
    return left;
  }

  private parseAnd(): SqlExpr {
    let left = this.parseNot();
    while (this.match(TokenType.AND)) {
      const right = this.parseNot();
      left = { kind: "binary", op: "and", left, right };
    }
    return left;
  }

  private parseNot(): SqlExpr {
    if (this.match(TokenType.NOT)) {
      // NOT EXISTS (SELECT ...)
      if (this.match(TokenType.EXISTS)) {
        this.expect(TokenType.LPAREN);
        const subquery = this.parseSelect();
        this.expect(TokenType.RPAREN);
        return { kind: "exists", subquery, negated: true };
      }
      const operand = this.parseNot();
      return { kind: "unary", op: "not", operand };
    }
    return this.parseComparison();
  }

  private parseComparison(): SqlExpr {
    let left = this.parseAddSub();

    // IS NULL / IS NOT NULL
    if (this.match(TokenType.IS)) {
      if (this.match(TokenType.NOT)) {
        this.expect(TokenType.NULL);
        return { kind: "unary", op: "is_not_null", operand: left };
      }
      this.expect(TokenType.NULL);
      return { kind: "unary", op: "is_null", operand: left };
    }

    // BETWEEN
    if (this.match(TokenType.BETWEEN)) {
      const low = this.parseAddSub();
      this.expect(TokenType.AND);
      const high = this.parseAddSub();
      return { kind: "between", expr: left, low, high };
    }

    // NOT BETWEEN / NOT IN / NOT LIKE
    if (this.check(TokenType.NOT)) {
      const savedPos = this.pos;
      this.advance();
      if (this.match(TokenType.BETWEEN)) {
        const low = this.parseAddSub();
        this.expect(TokenType.AND);
        const high = this.parseAddSub();
        return { kind: "unary", op: "not", operand: { kind: "between", expr: left, low, high } };
      }
      if (this.match(TokenType.IN)) {
        return this.parseInList(left, true);
      }
      if (this.match(TokenType.LIKE)) {
        const right = this.parseAddSub();
        return { kind: "unary", op: "not", operand: { kind: "binary", op: "like", left, right } };
      }
      // Not a postfix NOT — restore position
      this.pos = savedPos;
    }

    // IN
    if (this.match(TokenType.IN)) {
      return this.parseInList(left, false);
    }

    // LIKE
    if (this.match(TokenType.LIKE)) {
      const right = this.parseAddSub();
      return { kind: "binary", op: "like", left, right };
    }

    // NEAR (vector search: column NEAR [0.1, 0.2, ...])
    if (this.match(TokenType.NEAR)) {
      return this.parseNearExpr(left);
    }

    // Standard comparisons
    const opMap: [TokenType, BinaryOp][] = [
      [TokenType.EQ, "eq"], [TokenType.NE, "ne"],
      [TokenType.LT, "lt"], [TokenType.LE, "le"],
      [TokenType.GT, "gt"], [TokenType.GE, "ge"],
    ];
    for (const [tt, op] of opMap) {
      if (this.match(tt)) {
        const right = this.parseAddSub();
        return { kind: "binary", op, left, right };
      }
    }

    return left;
  }

  private parseInList(expr: SqlExpr, negated: boolean): SqlExpr {
    this.expect(TokenType.LPAREN);
    const values: SqlExpr[] = [];
    if (!this.check(TokenType.RPAREN)) {
      do {
        values.push(this.parseExpr());
      } while (this.match(TokenType.COMMA));
    }
    this.expect(TokenType.RPAREN);
    return { kind: "in_list", expr, values, negated };
  }

  private parseNearExpr(column: SqlExpr): SqlExpr {
    // NEAR [0.1, 0.2, ...] TOPK 10
    this.expect(TokenType.LBRACKET);
    const vector: number[] = [];
    if (!this.check(TokenType.RBRACKET)) {
      do {
        const neg = this.match(TokenType.MINUS);
        const tok = this.expect(TokenType.NUMBER);
        const num = parseFloat(tok.lexeme);
        vector.push(neg ? -num : num);
      } while (this.match(TokenType.COMMA));
    }
    this.expect(TokenType.RBRACKET);

    let topK: number | undefined;
    if (this.match(TokenType.TOPK)) {
      const tok = this.expect(TokenType.NUMBER);
      topK = parseInt(tok.lexeme, 10);
    }

    return { kind: "near", column, vector, topK };
  }

  private parseAddSub(): SqlExpr {
    let left = this.parseMulDiv();
    while (true) {
      if (this.match(TokenType.PLUS)) {
        left = { kind: "binary", op: "add", left, right: this.parseMulDiv() };
      } else if (this.match(TokenType.MINUS)) {
        left = { kind: "binary", op: "subtract", left, right: this.parseMulDiv() };
      } else if (this.match(TokenType.CONCAT)) {
        left = { kind: "binary", op: "concat", left, right: this.parseMulDiv() };
      } else {
        break;
      }
    }
    return left;
  }

  private parseMulDiv(): SqlExpr {
    let left = this.parseUnary();
    while (true) {
      if (this.match(TokenType.STAR)) {
        left = { kind: "binary", op: "multiply", left, right: this.parseUnary() };
      } else if (this.match(TokenType.SLASH)) {
        left = { kind: "binary", op: "divide", left, right: this.parseUnary() };
      } else {
        break;
      }
    }
    return left;
  }

  private parseUnary(): SqlExpr {
    if (this.match(TokenType.MINUS)) {
      return { kind: "unary", op: "minus", operand: this.parseUnary() };
    }
    return this.parsePrimary();
  }

  private parsePrimary(): SqlExpr {
    const tok = this.current();

    // NULL
    if (this.match(TokenType.NULL)) {
      return { kind: "value", value: { type: "null" } };
    }

    // Boolean TRUE/FALSE
    if (this.match(TokenType.TRUE)) {
      return { kind: "value", value: { type: "boolean", value: true } };
    }
    if (this.match(TokenType.FALSE)) {
      return { kind: "value", value: { type: "boolean", value: false } };
    }

    // Number
    if (this.match(TokenType.NUMBER)) {
      const num = tok.lexeme.includes(".") ? parseFloat(tok.lexeme) : parseInt(tok.lexeme, 10);
      const valType = tok.lexeme.includes(".") ? "float" as const : "integer" as const;
      return { kind: "value", value: { type: valType, value: num } };
    }

    // String literal
    if (this.match(TokenType.STRING)) {
      // Strip surrounding quotes
      const raw = tok.lexeme.slice(1, -1).replace(/''/g, "'").replace(/\\'/g, "'");
      return { kind: "value", value: { type: "string", value: raw } };
    }

    // Parameter binding (?)
    if (this.match(TokenType.PARAMETER)) {
      const index = this.paramCount++;
      return { kind: "parameter", index };
    }

    // EXISTS (SELECT ...)
    if (this.match(TokenType.EXISTS)) {
      this.expect(TokenType.LPAREN);
      const subquery = this.parseSelect();
      this.expect(TokenType.RPAREN);
      return { kind: "exists", subquery, negated: false };
    }

    // Parenthesized expression
    if (this.match(TokenType.LPAREN)) {
      const expr = this.parseExpr();
      this.expect(TokenType.RPAREN);
      return expr;
    }

    // CASE expression
    if (this.match(TokenType.CASE)) {
      return this.parseCaseExpr();
    }

    // CAST(expr AS type)
    if (this.match(TokenType.CAST)) {
      this.expect(TokenType.LPAREN);
      const expr = this.parseExpr();
      this.expect(TokenType.AS);
      const targetType = this.parseIdentifier();
      this.expect(TokenType.RPAREN);
      return { kind: "cast", expr, targetType };
    }

    // STAR (for SELECT *)
    if (this.match(TokenType.STAR)) {
      return { kind: "star" };
    }

    // Function call or column reference (including aggregate/window function keywords)
    if (tok.type === TokenType.IDENTIFIER || isFunctionKeyword(tok.type)) {
      const name = tok.lexeme;
      this.advance();

      // Check for table.column
      if (this.match(TokenType.DOT)) {
        const colTok = this.current();
        if (this.match(TokenType.STAR)) {
          // table.* in expression context
          return { kind: "column", table: name, name: "*" };
        }
        const colName = this.parseIdentifier();
        return { kind: "column", table: name, name: colName };
      }

      // Function call
      if (this.match(TokenType.LPAREN)) {
        return this.parseFunctionCall(name);
      }

      return { kind: "column", name };
    }

    throw this.error(`Unexpected token '${tok.lexeme}'`);
  }

  private parseFunctionCall(name: string): SqlExpr {
    let distinct = false;
    const args: SqlExpr[] = [];

    if (!this.check(TokenType.RPAREN)) {
      if (this.match(TokenType.DISTINCT)) {
        distinct = true;
      }
      // Handle COUNT(*)
      if (this.check(TokenType.STAR)) {
        this.advance();
        args.push({ kind: "star" });
      } else {
        do {
          args.push(this.parseExpr());
        } while (this.match(TokenType.COMMA));
      }
    }

    this.expect(TokenType.RPAREN);

    // Check for OVER (window function)
    let window: SqlWindowSpec | undefined;
    if (this.match(TokenType.OVER)) {
      window = this.parseWindowSpec();
    }

    return { kind: "call", name: name.toUpperCase(), args, distinct: distinct || undefined, window };
  }

  private parseWindowSpec(): SqlWindowSpec {
    this.expect(TokenType.LPAREN);
    const spec: SqlWindowSpec = {};

    if (this.match(TokenType.PARTITION)) {
      this.expect(TokenType.BY);
      spec.partitionBy = [];
      do {
        spec.partitionBy.push(this.parseIdentifier());
      } while (this.match(TokenType.COMMA));
    }

    if (this.check(TokenType.ORDER)) {
      spec.orderBy = this.parseOrderBy();
    }

    // Frame clause
    if (this.check(TokenType.ROWS) || this.check(TokenType.RANGE)) {
      const frameType = this.match(TokenType.ROWS) ? "rows" as const : (this.advance(), "range" as const);
      this.match(TokenType.BETWEEN); // consume optional BETWEEN keyword
      const start = this.parseFrameBound();
      let end: NonNullable<SqlWindowSpec["frame"]>["end"];
      if (this.match(TokenType.AND)) {
        end = this.parseFrameBound();
      }
      spec.frame = { type: frameType, start, end };
    }

    this.expect(TokenType.RPAREN);
    return spec;
  }

  private parseFrameBound(): NonNullable<NonNullable<SqlWindowSpec["frame"]>["start"]> {
    if (this.match(TokenType.UNBOUNDED)) {
      if (this.match(TokenType.PRECEDING)) return { type: "unbounded_preceding" };
      if (this.match(TokenType.FOLLOWING)) return { type: "unbounded_following" };
      throw this.error("Expected PRECEDING or FOLLOWING after UNBOUNDED");
    }
    if (this.match(TokenType.CURRENT)) {
      // Accept both "CURRENT ROW" and just "CURRENT"
      this.match(TokenType.IDENTIFIER); // consume ROW if present
      return { type: "current_row" };
    }
    // N PRECEDING / N FOLLOWING
    const tok = this.expect(TokenType.NUMBER);
    const offset = parseInt(tok.lexeme, 10);
    if (this.match(TokenType.PRECEDING)) return { type: "preceding", offset };
    if (this.match(TokenType.FOLLOWING)) return { type: "following", offset };
    throw this.error("Expected PRECEDING or FOLLOWING");
  }

  private parseCaseExpr(): SqlExpr {
    let operand: SqlExpr | undefined;
    // Simple CASE: CASE expr WHEN ...
    if (!this.check(TokenType.WHEN)) {
      operand = this.parseExpr();
    }

    const whenClauses: { condition: SqlExpr; result: SqlExpr }[] = [];
    while (this.match(TokenType.WHEN)) {
      const condition = this.parseExpr();
      this.expect(TokenType.THEN);
      const result = this.parseExpr();
      whenClauses.push({ condition, result });
    }

    let elseResult: SqlExpr | undefined;
    if (this.match(TokenType.ELSE)) {
      elseResult = this.parseExpr();
    }

    this.expect(TokenType.END);
    return { kind: "case_expr", operand, whenClauses, elseResult };
  }
}

function isFunctionKeyword(type: TokenType): boolean {
  return type === TokenType.COUNT || type === TokenType.SUM ||
    type === TokenType.AVG || type === TokenType.MIN || type === TokenType.MAX ||
    type === TokenType.ROW_NUMBER || type === TokenType.RANK ||
    type === TokenType.DENSE_RANK || type === TokenType.LAG || type === TokenType.LEAD;
}

function isKeywordIdentifier(type: TokenType): boolean {
  // Keywords that can appear as identifiers (column names, aliases)
  return type === TokenType.ASC || type === TokenType.DESC ||
    type === TokenType.LEFT || type === TokenType.RIGHT ||
    type === TokenType.FULL || type === TokenType.CROSS ||
    type === TokenType.INNER || type === TokenType.OUTER ||
    type === TokenType.ALL || type === TokenType.ROWS ||
    type === TokenType.RANGE || type === TokenType.CURRENT ||
    type === TokenType.COUNT || type === TokenType.SUM ||
    type === TokenType.AVG || type === TokenType.MIN || type === TokenType.MAX ||
    type === TokenType.RANK || type === TokenType.PARTITION ||
    type === TokenType.OVER || type === TokenType.ROW_NUMBER ||
    type === TokenType.DENSE_RANK || type === TokenType.LAG || type === TokenType.LEAD ||
    type === TokenType.NULL || type === TokenType.TRUE || type === TokenType.FALSE ||
    type === TokenType.DATA || type === TokenType.VERSION;
}

export class SqlParseError extends Error {
  position: number;
  constructor(message: string, position: number) {
    super(message);
    this.name = "SqlParseError";
    this.position = position;
  }
}
