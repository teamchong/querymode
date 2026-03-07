import { describe, it, expect } from "vitest";
import { parse, parseStatement } from "./parser.js";
import { SqlParseError } from "./parser.js";

describe("SQL Parser", () => {
  it("parses simple SELECT *", () => {
    const stmt = parse("SELECT * FROM users");
    expect(stmt.columns).toHaveLength(1);
    expect(stmt.columns[0].expr.kind).toBe("star");
    expect(stmt.from).toEqual({ kind: "simple", name: "users", alias: undefined });
  });

  it("parses SELECT with column list", () => {
    const stmt = parse("SELECT name, age FROM users");
    expect(stmt.columns).toHaveLength(2);
    expect(stmt.columns[0].expr).toEqual({ kind: "column", name: "name" });
    expect(stmt.columns[1].expr).toEqual({ kind: "column", name: "age" });
  });

  it("parses SELECT with alias", () => {
    const stmt = parse("SELECT name AS n FROM users");
    expect(stmt.columns[0].alias).toBe("n");
  });

  it("parses WHERE with AND", () => {
    const stmt = parse("SELECT * FROM t WHERE a > 1 AND b = 'x'");
    expect(stmt.where).toBeDefined();
    expect(stmt.where!.kind).toBe("binary");
    if (stmt.where!.kind === "binary") {
      expect(stmt.where!.op).toBe("and");
    }
  });

  it("parses WHERE with OR", () => {
    const stmt = parse("SELECT * FROM t WHERE a = 1 OR b = 2");
    expect(stmt.where!.kind).toBe("binary");
    if (stmt.where!.kind === "binary") {
      expect(stmt.where!.op).toBe("or");
    }
  });

  it("parses IS NULL / IS NOT NULL", () => {
    const stmt = parse("SELECT * FROM t WHERE a IS NULL AND b IS NOT NULL");
    expect(stmt.where!.kind).toBe("binary");
    if (stmt.where!.kind === "binary") {
      expect(stmt.where!.left.kind).toBe("unary");
      if (stmt.where!.left.kind === "unary") {
        expect(stmt.where!.left.op).toBe("is_null");
      }
      expect(stmt.where!.right.kind).toBe("unary");
      if (stmt.where!.right.kind === "unary") {
        expect(stmt.where!.right.op).toBe("is_not_null");
      }
    }
  });

  it("parses IN list", () => {
    const stmt = parse("SELECT * FROM t WHERE id IN (1, 2, 3)");
    expect(stmt.where!.kind).toBe("in_list");
    if (stmt.where!.kind === "in_list") {
      expect(stmt.where!.values).toHaveLength(3);
      expect(stmt.where!.negated).toBe(false);
    }
  });

  it("parses NOT IN list", () => {
    const stmt = parse("SELECT * FROM t WHERE id NOT IN (1, 2)");
    expect(stmt.where!.kind).toBe("in_list");
    if (stmt.where!.kind === "in_list") {
      expect(stmt.where!.negated).toBe(true);
    }
  });

  it("parses BETWEEN", () => {
    const stmt = parse("SELECT * FROM t WHERE age BETWEEN 18 AND 65");
    expect(stmt.where!.kind).toBe("between");
    if (stmt.where!.kind === "between") {
      expect(stmt.where!.low).toEqual({ kind: "value", value: { type: "integer", value: 18 } });
      expect(stmt.where!.high).toEqual({ kind: "value", value: { type: "integer", value: 65 } });
    }
  });

  it("parses GROUP BY", () => {
    const stmt = parse("SELECT region, COUNT(*) FROM sales GROUP BY region");
    expect(stmt.groupBy).toBeDefined();
    expect(stmt.groupBy!.columns).toEqual(["region"]);
  });

  it("parses GROUP BY with HAVING", () => {
    const stmt = parse("SELECT region, COUNT(*) FROM sales GROUP BY region HAVING COUNT(*) > 10");
    expect(stmt.groupBy!.having).toBeDefined();
  });

  it("parses ORDER BY", () => {
    const stmt = parse("SELECT * FROM t ORDER BY name ASC, age DESC");
    expect(stmt.orderBy).toHaveLength(2);
    expect(stmt.orderBy![0]).toEqual({ column: "name", direction: "asc" });
    expect(stmt.orderBy![1]).toEqual({ column: "age", direction: "desc" });
  });

  it("parses LIMIT and OFFSET", () => {
    const stmt = parse("SELECT * FROM t LIMIT 10 OFFSET 5");
    expect(stmt.limit).toBe(10);
    expect(stmt.offset).toBe(5);
  });

  it("parses DISTINCT", () => {
    const stmt = parse("SELECT DISTINCT name FROM users");
    expect(stmt.distinct).toBe(true);
  });

  it("parses aggregate functions", () => {
    const stmt = parse("SELECT COUNT(*), SUM(amount), AVG(price) FROM orders");
    expect(stmt.columns).toHaveLength(3);
    expect(stmt.columns[0].expr.kind).toBe("call");
    if (stmt.columns[0].expr.kind === "call") {
      expect(stmt.columns[0].expr.name).toBe("COUNT");
    }
  });

  it("parses COUNT(DISTINCT col)", () => {
    const stmt = parse("SELECT COUNT(DISTINCT name) FROM t");
    const col = stmt.columns[0].expr;
    expect(col.kind).toBe("call");
    if (col.kind === "call") {
      expect(col.distinct).toBe(true);
      expect(col.args[0]).toEqual({ kind: "column", name: "name" });
    }
  });

  it("parses INNER JOIN", () => {
    const stmt = parse("SELECT * FROM orders JOIN users ON orders.user_id = users.id");
    expect(stmt.from.kind).toBe("join");
    if (stmt.from.kind === "join") {
      expect(stmt.from.join.joinType).toBe("inner");
      expect(stmt.from.join.table).toEqual({ kind: "simple", name: "users", alias: undefined });
    }
  });

  it("parses LEFT JOIN", () => {
    const stmt = parse("SELECT * FROM orders LEFT JOIN users ON orders.uid = users.id");
    if (stmt.from.kind === "join") {
      expect(stmt.from.join.joinType).toBe("left");
    }
  });

  it("parses NEAR vector search", () => {
    const stmt = parse("SELECT * FROM items WHERE embedding NEAR [0.1, 0.2, 0.3] TOPK 5");
    expect(stmt.where!.kind).toBe("near");
    if (stmt.where!.kind === "near") {
      expect(stmt.where!.vector).toEqual([0.1, 0.2, 0.3]);
      expect(stmt.where!.topK).toBe(5);
    }
  });

  it("parses window function", () => {
    const stmt = parse("SELECT ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) AS rn FROM employees");
    const col = stmt.columns[0];
    expect(col.alias).toBe("rn");
    expect(col.expr.kind).toBe("call");
    if (col.expr.kind === "call") {
      expect(col.expr.name).toBe("ROW_NUMBER");
      expect(col.expr.window).toBeDefined();
      expect(col.expr.window!.partitionBy).toEqual(["dept"]);
      expect(col.expr.window!.orderBy).toEqual([{ column: "salary", direction: "desc" }]);
    }
  });

  it("parses UNION ALL", () => {
    const stmt = parse("SELECT a FROM t1 UNION ALL SELECT b FROM t2");
    expect(stmt.setOperation).toBeDefined();
    expect(stmt.setOperation!.opType).toBe("union_all");
    expect(stmt.setOperation!.right.from).toEqual({ kind: "simple", name: "t2", alias: undefined });
  });

  it("parses INTERSECT", () => {
    const stmt = parse("SELECT id FROM t1 INTERSECT SELECT id FROM t2");
    expect(stmt.setOperation!.opType).toBe("intersect");
  });

  it("parses CASE expression", () => {
    const stmt = parse("SELECT CASE WHEN age > 18 THEN 'adult' ELSE 'minor' END FROM users");
    const expr = stmt.columns[0].expr;
    expect(expr.kind).toBe("case_expr");
    if (expr.kind === "case_expr") {
      expect(expr.whenClauses).toHaveLength(1);
      expect(expr.elseResult).toBeDefined();
    }
  });

  it("parses CAST expression", () => {
    const stmt = parse("SELECT CAST(age AS float) FROM users");
    const expr = stmt.columns[0].expr;
    expect(expr.kind).toBe("cast");
    if (expr.kind === "cast") {
      expect(expr.targetType).toBe("float");
    }
  });

  it("parses table alias", () => {
    const stmt = parse("SELECT * FROM users u");
    expect(stmt.from).toEqual({ kind: "simple", name: "users", alias: "u" });
  });

  it("parses trailing semicolon", () => {
    const stmt = parse("SELECT * FROM t;");
    expect(stmt.from).toEqual({ kind: "simple", name: "t", alias: undefined });
  });

  it("throws on malformed SQL", () => {
    expect(() => parse("SELEC * FROM t")).toThrow(SqlParseError);
  });

  it("throws on missing FROM", () => {
    expect(() => parse("SELECT *")).toThrow(SqlParseError);
  });

  it("parses identifier after FROM as table alias", () => {
    // "GARBAGE" is a valid table alias in SQL
    const stmt = parse("SELECT * FROM t GARBAGE");
    expect(stmt.from).toEqual({ kind: "simple", name: "t", alias: "GARBAGE" });
  });

  it("throws on truly invalid trailing tokens", () => {
    expect(() => parse("SELECT * FROM t WHERE x = 1 SELECT")).toThrow(SqlParseError);
  });

  it("parses LIKE", () => {
    const stmt = parse("SELECT * FROM t WHERE name LIKE '%alice%'");
    expect(stmt.where!.kind).toBe("binary");
    if (stmt.where!.kind === "binary") {
      expect(stmt.where!.op).toBe("like");
    }
  });

  it("parses negative numbers", () => {
    const stmt = parse("SELECT * FROM t WHERE x > -5");
    if (stmt.where!.kind === "binary") {
      expect(stmt.where!.right.kind).toBe("unary");
    }
  });

  it("parses parenthesized expressions", () => {
    const stmt = parse("SELECT * FROM t WHERE (a = 1 OR b = 2) AND c = 3");
    expect(stmt.where!.kind).toBe("binary");
    if (stmt.where!.kind === "binary") {
      expect(stmt.where!.op).toBe("and");
      expect(stmt.where!.left.kind).toBe("binary");
      if (stmt.where!.left.kind === "binary") {
        expect(stmt.where!.left.op).toBe("or");
      }
    }
  });

  it("parses EXISTS subquery", () => {
    const stmt = parse("SELECT * FROM t WHERE EXISTS (SELECT id FROM u WHERE u.tid = t.id)");
    expect(stmt.where!.kind).toBe("exists");
    if (stmt.where!.kind === "exists") {
      expect(stmt.where!.negated).toBe(false);
      expect(stmt.where!.subquery.from).toEqual({ kind: "simple", name: "u", alias: undefined });
    }
  });

  it("parses NOT EXISTS subquery", () => {
    const stmt = parse("SELECT * FROM t WHERE NOT EXISTS (SELECT id FROM u WHERE u.tid = t.id)");
    expect(stmt.where!.kind).toBe("exists");
    if (stmt.where!.kind === "exists") {
      expect(stmt.where!.negated).toBe(true);
      expect(stmt.where!.subquery.from).toEqual({ kind: "simple", name: "u", alias: undefined });
    }
  });

  it("parses NATURAL JOIN", () => {
    const stmt = parse("SELECT * FROM a NATURAL JOIN b");
    expect(stmt.from.kind).toBe("join");
    if (stmt.from.kind === "join") {
      expect(stmt.from.join.natural).toBe(true);
      expect(stmt.from.join.joinType).toBe("inner");
      expect(stmt.from.join.table).toEqual({ kind: "simple", name: "b", alias: undefined });
    }
  });

  it("parses NATURAL LEFT JOIN", () => {
    const stmt = parse("SELECT * FROM a NATURAL LEFT JOIN b");
    expect(stmt.from.kind).toBe("join");
    if (stmt.from.kind === "join") {
      expect(stmt.from.join.natural).toBe(true);
      expect(stmt.from.join.joinType).toBe("left");
    }
  });

  it("parses JOIN with USING clause", () => {
    const stmt = parse("SELECT * FROM a JOIN b USING (id)");
    expect(stmt.from.kind).toBe("join");
    if (stmt.from.kind === "join") {
      expect(stmt.from.join.using).toEqual(["id"]);
      expect(stmt.from.join.onCondition).toBeUndefined();
    }
  });

  it("parses JOIN with USING clause with multiple columns", () => {
    const stmt = parse("SELECT * FROM a JOIN b USING (id, name)");
    expect(stmt.from.kind).toBe("join");
    if (stmt.from.kind === "join") {
      expect(stmt.from.join.using).toEqual(["id", "name"]);
    }
  });

  it("parses ? parameter binding", () => {
    const stmt = parse("SELECT * FROM t WHERE id = ?");
    expect(stmt.where!.kind).toBe("binary");
    if (stmt.where!.kind === "binary") {
      expect(stmt.where!.right.kind).toBe("parameter");
      if (stmt.where!.right.kind === "parameter") {
        expect(stmt.where!.right.index).toBe(0);
      }
    }
  });

  it("parses multiple ? parameters with incrementing indices", () => {
    const stmt = parse("SELECT * FROM t WHERE a = ? AND b = ?");
    if (stmt.where!.kind === "binary" && stmt.where!.op === "and") {
      const left = stmt.where!.left;
      const right = stmt.where!.right;
      if (left.kind === "binary" && left.right.kind === "parameter") {
        expect(left.right.index).toBe(0);
      }
      if (right.kind === "binary" && right.right.kind === "parameter") {
        expect(right.right.index).toBe(1);
      }
    }
  });
});

describe("SQL Statement Parser", () => {
  it("parses SHOW VERSIONS FOR table", () => {
    const result = parseStatement("SHOW VERSIONS FOR users");
    expect(result.kind).toBe("show_versions");
    if (result.kind === "show_versions") {
      expect(result.stmt.table).toBe("users");
      expect(result.stmt.limit).toBeUndefined();
    }
  });

  it("parses SHOW VERSIONS FOR table LIMIT", () => {
    const result = parseStatement("SHOW VERSIONS FOR users LIMIT 10");
    if (result.kind === "show_versions") {
      expect(result.stmt.table).toBe("users");
      expect(result.stmt.limit).toBe(10);
    }
  });

  it("parses DIFF table VERSION n", () => {
    const result = parseStatement("DIFF users VERSION 3");
    expect(result.kind).toBe("diff");
    if (result.kind === "diff") {
      expect(result.stmt.table).toBe("users");
      expect(result.stmt.fromVersion).toBe(3);
      expect(result.stmt.toVersion).toBeUndefined();
    }
  });

  it("parses DIFF table VERSION n AND VERSION m", () => {
    const result = parseStatement("DIFF users VERSION 2 AND VERSION 5");
    if (result.kind === "diff") {
      expect(result.stmt.table).toBe("users");
      expect(result.stmt.fromVersion).toBe(2);
      expect(result.stmt.toVersion).toBe(5);
    }
  });

  it("parses DIFF with LIMIT", () => {
    const result = parseStatement("DIFF users VERSION 1 AND VERSION 3 LIMIT 50");
    if (result.kind === "diff") {
      expect(result.stmt.fromVersion).toBe(1);
      expect(result.stmt.toVersion).toBe(3);
      expect(result.stmt.limit).toBe(50);
    }
  });

  it("parses SELECT via parseStatement", () => {
    const result = parseStatement("SELECT * FROM t");
    expect(result.kind).toBe("select");
    if (result.kind === "select") {
      expect(result.stmt.from).toEqual({ kind: "simple", name: "t", alias: undefined });
    }
  });
});
