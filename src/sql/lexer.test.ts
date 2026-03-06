import { describe, it, expect } from "vitest";
import { tokenize, TokenType, SqlLexerError } from "./lexer.js";

describe("SQL Lexer", () => {
  it("tokenizes basic SELECT", () => {
    const tokens = tokenize("SELECT * FROM users WHERE id = 42");
    expect(tokens[0].type).toBe(TokenType.SELECT);
    expect(tokens[1].type).toBe(TokenType.STAR);
    expect(tokens[2].type).toBe(TokenType.FROM);
    expect(tokens[3]).toMatchObject({ type: TokenType.IDENTIFIER, lexeme: "users" });
    expect(tokens[4].type).toBe(TokenType.WHERE);
    expect(tokens[5]).toMatchObject({ type: TokenType.IDENTIFIER, lexeme: "id" });
    expect(tokens[6].type).toBe(TokenType.EQ);
    expect(tokens[7]).toMatchObject({ type: TokenType.NUMBER, lexeme: "42" });
    expect(tokens[8].type).toBe(TokenType.EOF);
  });

  it("tokenizes string literals", () => {
    const tokens = tokenize("SELECT 'hello world' FROM t");
    expect(tokens[1]).toMatchObject({ type: TokenType.STRING, lexeme: "'hello world'" });
  });

  it("tokenizes comparison operators", () => {
    const tokens = tokenize("a != b AND c <> d AND e <= f AND g >= h");
    expect(tokens[1].type).toBe(TokenType.NE);
    expect(tokens[5].type).toBe(TokenType.NE);
    expect(tokens[9].type).toBe(TokenType.LE);
    expect(tokens[13].type).toBe(TokenType.GE);
  });

  it("tokenizes float numbers", () => {
    const tokens = tokenize("SELECT 3.14");
    expect(tokens[1]).toMatchObject({ type: TokenType.NUMBER, lexeme: "3.14" });
  });

  it("is case-insensitive for keywords", () => {
    const tokens = tokenize("select FROM Where");
    expect(tokens[0].type).toBe(TokenType.SELECT);
    expect(tokens[1].type).toBe(TokenType.FROM);
    expect(tokens[2].type).toBe(TokenType.WHERE);
  });

  it("tokenizes NEAR and TOPK", () => {
    const tokens = tokenize("NEAR TOPK");
    expect(tokens[0].type).toBe(TokenType.NEAR);
    expect(tokens[1].type).toBe(TokenType.TOPK);
  });

  it("tokenizes brackets for vector literals", () => {
    const tokens = tokenize("[0.1, 0.2]");
    expect(tokens[0].type).toBe(TokenType.LBRACKET);
    expect(tokens[1].type).toBe(TokenType.NUMBER);
    expect(tokens[2].type).toBe(TokenType.COMMA);
    expect(tokens[3].type).toBe(TokenType.NUMBER);
    expect(tokens[4].type).toBe(TokenType.RBRACKET);
  });

  it("tokenizes quoted identifiers", () => {
    const tokens = tokenize('"my column" `another`');
    expect(tokens[0]).toMatchObject({ type: TokenType.IDENTIFIER, lexeme: '"my column"' });
    expect(tokens[1]).toMatchObject({ type: TokenType.IDENTIFIER, lexeme: '`another`' });
  });

  it("tokenizes concat operator", () => {
    const tokens = tokenize("a || b");
    expect(tokens[1].type).toBe(TokenType.CONCAT);
  });

  it("reports position on error", () => {
    expect(() => tokenize("SELECT @")).toThrow(SqlLexerError);
    try {
      tokenize("SELECT @");
    } catch (e) {
      expect((e as SqlLexerError).position).toBe(7);
    }
  });

  it("tokenizes aggregate function keywords", () => {
    const tokens = tokenize("COUNT SUM AVG MIN MAX");
    expect(tokens[0].type).toBe(TokenType.COUNT);
    expect(tokens[1].type).toBe(TokenType.SUM);
    expect(tokens[2].type).toBe(TokenType.AVG);
    expect(tokens[3].type).toBe(TokenType.MIN);
    expect(tokens[4].type).toBe(TokenType.MAX);
  });

  it("tokenizes window function keywords", () => {
    const tokens = tokenize("ROW_NUMBER RANK DENSE_RANK OVER PARTITION");
    expect(tokens[0].type).toBe(TokenType.ROW_NUMBER);
    expect(tokens[1].type).toBe(TokenType.RANK);
    expect(tokens[2].type).toBe(TokenType.DENSE_RANK);
    expect(tokens[3].type).toBe(TokenType.OVER);
    expect(tokens[4].type).toBe(TokenType.PARTITION);
  });

  it("tokenizes boolean literals", () => {
    const tokens = tokenize("TRUE FALSE");
    expect(tokens[0].type).toBe(TokenType.TRUE);
    expect(tokens[1].type).toBe(TokenType.FALSE);
  });

  it("tokenizes semicolons", () => {
    const tokens = tokenize("SELECT 1;");
    expect(tokens[2].type).toBe(TokenType.SEMICOLON);
  });

  it("throws on unterminated string literal", () => {
    expect(() => tokenize("SELECT 'hello")).toThrow(SqlLexerError);
    expect(() => tokenize("SELECT 'hello")).toThrow(/Unterminated string literal/);
  });

  it("throws on unterminated double-quoted identifier", () => {
    expect(() => tokenize('SELECT "unclosed')).toThrow(SqlLexerError);
    expect(() => tokenize('SELECT "unclosed')).toThrow(/Unterminated quoted identifier/);
  });

  it("throws on unterminated backtick identifier", () => {
    expect(() => tokenize("SELECT `unclosed")).toThrow(SqlLexerError);
    expect(() => tokenize("SELECT `unclosed")).toThrow(/Unterminated backtick identifier/);
  });
});
