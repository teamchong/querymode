/** SQL Lexer — ported from lanceql/src/sql/lexer.zig */

export enum TokenType {
  // Keywords
  SELECT, FROM, WHERE, AND, OR, NOT, IN, BETWEEN, LIKE,
  IS, NULL, AS, DISTINCT, ORDER, BY, ASC, DESC, LIMIT, OFFSET,
  GROUP, HAVING,

  // DDL keywords
  CREATE, DROP, ALTER, TABLE, INDEX, VECTOR, IF, SHOW, INDEXES,

  // Time travel / diff keywords
  DIFF, VERSION, VERSIONS, CHANGES, SINCE, FOR, HEAD,

  // JOIN keywords
  JOIN, LEFT, RIGHT, INNER, OUTER, FULL, CROSS, ON, NATURAL, USING,

  // Set operations
  UNION, INTERSECT, EXCEPT, ALL,

  // CASE
  CASE, WHEN, THEN, ELSE, END,

  // Subquery keywords
  EXISTS,

  // Type casting
  CAST,

  // Window functions
  OVER, PARTITION, ROWS, RANGE, UNBOUNDED, PRECEDING, FOLLOWING, CURRENT,

  // Vector search
  NEAR, TOPK,

  // Logic table extension
  WITH, DATA, LOGIC_TABLE, LOGIC,

  // Aggregate functions (recognized as keywords for cleaner parsing)
  COUNT, SUM, AVG, MIN, MAX,

  // Ranking functions
  ROW_NUMBER, RANK, DENSE_RANK, NTILE, PERCENT_RANK, CUME_DIST,

  // Offset/Analytic functions
  LAG, LEAD, FIRST_VALUE, LAST_VALUE, NTH_VALUE,

  // Time window functions
  INTERVAL, SESSION, TUMBLE, HOP,

  // Boolean
  TRUE, FALSE,

  // Literals
  IDENTIFIER, NUMBER, STRING,

  // Operators
  STAR, EQ, NE, LT, LE, GT, GE,
  PLUS, MINUS, SLASH, CONCAT,
  COMMA, DOT, LPAREN, RPAREN, SEMICOLON, LBRACKET, RBRACKET,
  PARAMETER,

  EOF,
}

export interface Token {
  type: TokenType;
  lexeme: string;
  position: number;
}

const KEYWORDS = new Map<string, TokenType>([
  ["SELECT", TokenType.SELECT], ["FROM", TokenType.FROM], ["WHERE", TokenType.WHERE],
  ["AND", TokenType.AND], ["OR", TokenType.OR], ["NOT", TokenType.NOT],
  ["IN", TokenType.IN], ["BETWEEN", TokenType.BETWEEN], ["LIKE", TokenType.LIKE],
  ["IS", TokenType.IS], ["NULL", TokenType.NULL], ["AS", TokenType.AS],
  ["DISTINCT", TokenType.DISTINCT], ["ORDER", TokenType.ORDER], ["BY", TokenType.BY],
  ["ASC", TokenType.ASC], ["DESC", TokenType.DESC], ["LIMIT", TokenType.LIMIT],
  ["OFFSET", TokenType.OFFSET], ["GROUP", TokenType.GROUP], ["HAVING", TokenType.HAVING],
  ["CREATE", TokenType.CREATE], ["DROP", TokenType.DROP], ["ALTER", TokenType.ALTER],
  ["TABLE", TokenType.TABLE], ["INDEX", TokenType.INDEX], ["VECTOR", TokenType.VECTOR],
  ["IF", TokenType.IF], ["SHOW", TokenType.SHOW], ["INDEXES", TokenType.INDEXES],
  ["DIFF", TokenType.DIFF], ["VERSION", TokenType.VERSION], ["VERSIONS", TokenType.VERSIONS],
  ["CHANGES", TokenType.CHANGES], ["SINCE", TokenType.SINCE], ["FOR", TokenType.FOR],
  ["HEAD", TokenType.HEAD],
  ["JOIN", TokenType.JOIN], ["LEFT", TokenType.LEFT], ["RIGHT", TokenType.RIGHT],
  ["INNER", TokenType.INNER], ["OUTER", TokenType.OUTER], ["FULL", TokenType.FULL],
  ["CROSS", TokenType.CROSS], ["ON", TokenType.ON], ["NATURAL", TokenType.NATURAL],
  ["USING", TokenType.USING],
  ["UNION", TokenType.UNION], ["INTERSECT", TokenType.INTERSECT],
  ["EXCEPT", TokenType.EXCEPT], ["ALL", TokenType.ALL],
  ["CASE", TokenType.CASE], ["WHEN", TokenType.WHEN], ["THEN", TokenType.THEN],
  ["ELSE", TokenType.ELSE], ["END", TokenType.END],
  ["EXISTS", TokenType.EXISTS],
  ["CAST", TokenType.CAST],
  ["OVER", TokenType.OVER], ["PARTITION", TokenType.PARTITION],
  ["ROWS", TokenType.ROWS], ["RANGE", TokenType.RANGE],
  ["UNBOUNDED", TokenType.UNBOUNDED], ["PRECEDING", TokenType.PRECEDING],
  ["FOLLOWING", TokenType.FOLLOWING], ["CURRENT", TokenType.CURRENT],
  ["NEAR", TokenType.NEAR], ["TOPK", TokenType.TOPK],
  ["WITH", TokenType.WITH], ["DATA", TokenType.DATA],
  ["LOGIC_TABLE", TokenType.LOGIC_TABLE], ["LOGIC", TokenType.LOGIC],
  ["COUNT", TokenType.COUNT], ["SUM", TokenType.SUM], ["AVG", TokenType.AVG],
  ["MIN", TokenType.MIN], ["MAX", TokenType.MAX],
  ["ROW_NUMBER", TokenType.ROW_NUMBER], ["RANK", TokenType.RANK],
  ["DENSE_RANK", TokenType.DENSE_RANK], ["NTILE", TokenType.NTILE],
  ["PERCENT_RANK", TokenType.PERCENT_RANK], ["CUME_DIST", TokenType.CUME_DIST],
  ["LAG", TokenType.LAG], ["LEAD", TokenType.LEAD],
  ["FIRST_VALUE", TokenType.FIRST_VALUE], ["LAST_VALUE", TokenType.LAST_VALUE],
  ["NTH_VALUE", TokenType.NTH_VALUE],
  ["INTERVAL", TokenType.INTERVAL], ["SESSION", TokenType.SESSION],
  ["TUMBLE", TokenType.TUMBLE], ["HOP", TokenType.HOP],
  ["TRUE", TokenType.TRUE], ["FALSE", TokenType.FALSE],
]);

export function tokenize(sql: string): Token[] {
  const tokens: Token[] = [];
  let pos = 0;
  const len = sql.length;

  while (pos < len) {
    // Skip whitespace
    while (pos < len && /\s/.test(sql[pos])) pos++;
    if (pos >= len) break;

    const start = pos;
    const ch = sql[pos];

    // Line comments (--)
    if (ch === "-" && pos + 1 < len && sql[pos + 1] === "-") {
      while (pos < len && sql[pos] !== "\n") pos++;
      continue;
    }
    // Block comments (/* */)
    if (ch === "/" && pos + 1 < len && sql[pos + 1] === "*") {
      pos += 2;
      while (pos + 1 < len && !(sql[pos] === "*" && sql[pos + 1] === "/")) pos++;
      pos += 2;
      continue;
    }

    // Identifiers and keywords
    if (/[a-zA-Z_]/.test(ch)) {
      while (pos < len && /[a-zA-Z0-9_]/.test(sql[pos])) pos++;
      const lexeme = sql.slice(start, pos);
      const kwType = KEYWORDS.get(lexeme.toUpperCase());
      tokens.push({ type: kwType ?? TokenType.IDENTIFIER, lexeme, position: start });
      continue;
    }

    // Numbers (supports decimals and scientific notation: 1e10, 1.5e-3, 3E+5)
    if (/[0-9]/.test(ch)) {
      while (pos < len && /[0-9]/.test(sql[pos])) pos++;
      if (pos < len && sql[pos] === ".") {
        pos++;
        while (pos < len && /[0-9]/.test(sql[pos])) pos++;
      }
      if (pos < len && (sql[pos] === "e" || sql[pos] === "E")) {
        const ePos = pos;
        pos++;
        if (pos < len && (sql[pos] === "+" || sql[pos] === "-")) pos++;
        if (pos < len && /[0-9]/.test(sql[pos])) {
          while (pos < len && /[0-9]/.test(sql[pos])) pos++;
        } else {
          pos = ePos; // rollback — 'e' is not part of this number
        }
      }
      tokens.push({ type: TokenType.NUMBER, lexeme: sql.slice(start, pos), position: start });
      continue;
    }

    // String literals (single-quoted, supports '' and \' escaping)
    if (ch === "'") {
      pos++;
      while (pos < len) {
        if (sql[pos] === "'" && pos + 1 < len && sql[pos + 1] === "'") { pos += 2; continue; } // doubled quote
        if (sql[pos] === "'") break;
        if (sql[pos] === "\\" && pos + 1 < len) pos++; // backslash escape
        pos++;
      }
      if (pos >= len) throw new SqlLexerError(`Unterminated string literal starting at position ${start}`, start);
      pos++; // closing quote
      tokens.push({ type: TokenType.STRING, lexeme: sql.slice(start, pos), position: start });
      continue;
    }

    // Quoted identifiers (double-quoted)
    if (ch === '"') {
      pos++;
      while (pos < len && sql[pos] !== '"') pos++;
      if (pos >= len) throw new SqlLexerError(`Unterminated quoted identifier starting at position ${start}`, start);
      pos++; // closing quote
      tokens.push({ type: TokenType.IDENTIFIER, lexeme: sql.slice(start, pos), position: start });
      continue;
    }

    // Backtick-quoted identifiers
    if (ch === "`") {
      pos++;
      while (pos < len && sql[pos] !== "`") pos++;
      if (pos >= len) throw new SqlLexerError(`Unterminated backtick identifier starting at position ${start}`, start);
      pos++; // closing backtick
      tokens.push({ type: TokenType.IDENTIFIER, lexeme: sql.slice(start, pos), position: start });
      continue;
    }

    // Operators and punctuation
    pos++;
    switch (ch) {
      case "*": tokens.push({ type: TokenType.STAR, lexeme: "*", position: start }); break;
      case ",": tokens.push({ type: TokenType.COMMA, lexeme: ",", position: start }); break;
      case ".": tokens.push({ type: TokenType.DOT, lexeme: ".", position: start }); break;
      case "(": tokens.push({ type: TokenType.LPAREN, lexeme: "(", position: start }); break;
      case ")": tokens.push({ type: TokenType.RPAREN, lexeme: ")", position: start }); break;
      case ";": tokens.push({ type: TokenType.SEMICOLON, lexeme: ";", position: start }); break;
      case "+": tokens.push({ type: TokenType.PLUS, lexeme: "+", position: start }); break;
      case "-": tokens.push({ type: TokenType.MINUS, lexeme: "-", position: start }); break;
      case "/": tokens.push({ type: TokenType.SLASH, lexeme: "/", position: start }); break;
      case "[": tokens.push({ type: TokenType.LBRACKET, lexeme: "[", position: start }); break;
      case "]": tokens.push({ type: TokenType.RBRACKET, lexeme: "]", position: start }); break;
      case "?": tokens.push({ type: TokenType.PARAMETER, lexeme: "?", position: start }); break;
      case "=": tokens.push({ type: TokenType.EQ, lexeme: "=", position: start }); break;
      case "!":
        if (pos < len && sql[pos] === "=") {
          pos++;
          tokens.push({ type: TokenType.NE, lexeme: "!=", position: start });
        } else {
          throw new SqlLexerError(`Unexpected character '!' at position ${start}`, start);
        }
        break;
      case "<":
        if (pos < len && sql[pos] === "=") {
          pos++;
          tokens.push({ type: TokenType.LE, lexeme: "<=", position: start });
        } else if (pos < len && sql[pos] === ">") {
          pos++;
          tokens.push({ type: TokenType.NE, lexeme: "<>", position: start });
        } else {
          tokens.push({ type: TokenType.LT, lexeme: "<", position: start });
        }
        break;
      case ">":
        if (pos < len && sql[pos] === "=") {
          pos++;
          tokens.push({ type: TokenType.GE, lexeme: ">=", position: start });
        } else {
          tokens.push({ type: TokenType.GT, lexeme: ">", position: start });
        }
        break;
      case "|":
        if (pos < len && sql[pos] === "|") {
          pos++;
          tokens.push({ type: TokenType.CONCAT, lexeme: "||", position: start });
        } else {
          throw new SqlLexerError(`Unexpected character '|' at position ${start}`, start);
        }
        break;
      default:
        throw new SqlLexerError(`Unexpected character '${ch}' at position ${start}`, start);
    }
  }

  tokens.push({ type: TokenType.EOF, lexeme: "", position: pos });
  return tokens;
}

export class SqlLexerError extends Error {
  position: number;
  constructor(message: string, position: number) {
    super(message);
    this.name = "SqlLexerError";
    this.position = position;
  }
}
