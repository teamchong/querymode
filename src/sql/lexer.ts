/** SQL Lexer — ported from lanceql/src/sql/lexer.zig */

export enum TokenType {
  // Keywords
  SELECT, FROM, WHERE, AND, OR, NOT, IN, BETWEEN, LIKE,
  IS, NULL, AS, DISTINCT, ORDER, BY, ASC, DESC, LIMIT, OFFSET,
  GROUP, HAVING,

  // JOIN keywords
  JOIN, LEFT, RIGHT, INNER, OUTER, FULL, CROSS, ON,

  // Set operations
  UNION, INTERSECT, EXCEPT, ALL,

  // CASE
  CASE, WHEN, THEN, ELSE, END,

  // Type casting
  CAST,

  // Window functions
  OVER, PARTITION, ROWS, RANGE, UNBOUNDED, PRECEDING, FOLLOWING, CURRENT,

  // Vector search
  NEAR, TOPK,

  // Aggregate functions (recognized as keywords for cleaner parsing)
  COUNT, SUM, AVG, MIN, MAX,

  // Window function names
  ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD,

  // Boolean
  TRUE, FALSE,

  // Literals
  IDENTIFIER, NUMBER, STRING,

  // Operators
  STAR, EQ, NE, LT, LE, GT, GE,
  PLUS, MINUS, SLASH, CONCAT,
  COMMA, DOT, LPAREN, RPAREN, SEMICOLON, LBRACKET, RBRACKET,

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
  ["JOIN", TokenType.JOIN], ["LEFT", TokenType.LEFT], ["RIGHT", TokenType.RIGHT],
  ["INNER", TokenType.INNER], ["OUTER", TokenType.OUTER], ["FULL", TokenType.FULL],
  ["CROSS", TokenType.CROSS], ["ON", TokenType.ON],
  ["UNION", TokenType.UNION], ["INTERSECT", TokenType.INTERSECT],
  ["EXCEPT", TokenType.EXCEPT], ["ALL", TokenType.ALL],
  ["CASE", TokenType.CASE], ["WHEN", TokenType.WHEN], ["THEN", TokenType.THEN],
  ["ELSE", TokenType.ELSE], ["END", TokenType.END],
  ["CAST", TokenType.CAST],
  ["OVER", TokenType.OVER], ["PARTITION", TokenType.PARTITION],
  ["ROWS", TokenType.ROWS], ["RANGE", TokenType.RANGE],
  ["UNBOUNDED", TokenType.UNBOUNDED], ["PRECEDING", TokenType.PRECEDING],
  ["FOLLOWING", TokenType.FOLLOWING], ["CURRENT", TokenType.CURRENT],
  ["NEAR", TokenType.NEAR], ["TOPK", TokenType.TOPK],
  ["COUNT", TokenType.COUNT], ["SUM", TokenType.SUM], ["AVG", TokenType.AVG],
  ["MIN", TokenType.MIN], ["MAX", TokenType.MAX],
  ["ROW_NUMBER", TokenType.ROW_NUMBER], ["RANK", TokenType.RANK],
  ["DENSE_RANK", TokenType.DENSE_RANK], ["LAG", TokenType.LAG], ["LEAD", TokenType.LEAD],
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

    // Identifiers and keywords
    if (/[a-zA-Z_]/.test(ch)) {
      while (pos < len && /[a-zA-Z0-9_]/.test(sql[pos])) pos++;
      const lexeme = sql.slice(start, pos);
      const kwType = KEYWORDS.get(lexeme.toUpperCase());
      tokens.push({ type: kwType ?? TokenType.IDENTIFIER, lexeme, position: start });
      continue;
    }

    // Numbers
    if (/[0-9]/.test(ch)) {
      while (pos < len && /[0-9]/.test(sql[pos])) pos++;
      if (pos < len && sql[pos] === ".") {
        pos++;
        while (pos < len && /[0-9]/.test(sql[pos])) pos++;
      }
      tokens.push({ type: TokenType.NUMBER, lexeme: sql.slice(start, pos), position: start });
      continue;
    }

    // String literals (single-quoted)
    if (ch === "'") {
      pos++;
      while (pos < len && sql[pos] !== "'") {
        if (sql[pos] === "\\" && pos + 1 < len) pos++; // skip escape
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
