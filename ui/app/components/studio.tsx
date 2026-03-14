"use client";

import { useState, useCallback } from "react";
import { SchemaPanel, type TableSchema } from "./schema-panel";
import { CodeEditor } from "./editor";
import { ResultsPanel, type QueryResult } from "./results-panel";
import { SourceManager } from "./source-manager";

type Tab = "query" | "sources";

export function Studio() {
  const [tab, setTab] = useState<Tab>("query");
  const [schemas, setSchemas] = useState<TableSchema[]>([]);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [code, setCode] = useState(EXAMPLE_CODE);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadSchemas = useCallback(async () => {
    try {
      const res = await fetch("/api/schema");
      if (res.ok) setSchemas(await res.json());
    } catch {}
  }, []);

  const executeCode = useCallback(async () => {
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });
      if (!res.ok) {
        setError(await res.text());
        return;
      }
      setResult(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }, [code]);

  const insertSnippet = useCallback((snippet: string) => {
    setCode((prev) => prev + "\n" + snippet);
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* Header */}
      <header style={headerStyle}>
        <span style={{ fontWeight: 700, fontSize: "0.9rem" }}>QueryMode Studio</span>
        <nav style={{ display: "flex", gap: "1rem" }}>
          <TabButton active={tab === "query"} onClick={() => setTab("query")}>Query</TabButton>
          <TabButton active={tab === "sources"} onClick={() => setTab("sources")}>Sources</TabButton>
        </nav>
        <div style={{ flex: 1 }} />
        {tab === "query" && (
          <button onClick={executeCode} disabled={running} style={runBtnStyle}>
            {running ? "Running..." : "Run ▶"}
          </button>
        )}
      </header>

      {tab === "query" ? (
        <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
          {/* Schema sidebar */}
          <SchemaPanel schemas={schemas} onRefresh={loadSchemas} onInsert={insertSnippet} />

          {/* Editor + Results */}
          <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{ flex: 1, minHeight: 200, overflow: "auto", borderBottom: "1px solid #21262d" }}>
              <CodeEditor value={code} onChange={setCode} onRun={executeCode} />
            </div>
            <div style={{ flex: 1, minHeight: 200, overflow: "auto" }}>
              <ResultsPanel result={result} error={error} running={running} />
            </div>
          </div>
        </div>
      ) : (
        <SourceManager />
      )}
    </div>
  );
}

function TabButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: "none",
        border: "none",
        color: active ? "#58a6ff" : "#8b949e",
        cursor: "pointer",
        fontSize: "0.85rem",
        borderBottom: active ? "2px solid #58a6ff" : "2px solid transparent",
        padding: "0.25rem 0",
      }}
    >
      {children}
    </button>
  );
}

const headerStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: "1.5rem",
  padding: "0.5rem 1rem",
  borderBottom: "1px solid #21262d",
  background: "#161b22",
};

const runBtnStyle: React.CSSProperties = {
  background: "#238636",
  color: "white",
  border: "none",
  borderRadius: 6,
  padding: "0.35rem 1rem",
  cursor: "pointer",
  fontSize: "0.85rem",
  fontWeight: 600,
};

const EXAMPLE_CODE = `// QueryMode — write DataFrame or SQL queries
const df = qm.table("users")
  .filter("age", "gte", 18)
  .select("name", "email", "age")
  .sort("age", "desc")
  .limit(50)
  .execute();

// Or use SQL:
// qm.sql("SELECT name, email FROM users WHERE age >= 18 ORDER BY age DESC LIMIT 50")
`;
