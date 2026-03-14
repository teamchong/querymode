"use client";

export interface QueryResult {
  rows: Record<string, unknown>[];
  columns: string[];
  rowCount: number;
  bytesRead: number;
  pagesSkipped: number;
  durationMs: number;
  code?: string;
}

interface Props {
  result: QueryResult | null;
  error: string | null;
  running: boolean;
}

export function ResultsPanel({ result, error, running }: Props) {
  if (running) {
    return <StatusBar>Running query...</StatusBar>;
  }

  if (error) {
    return (
      <div style={{ padding: "1rem" }}>
        <StatusBar>Error</StatusBar>
        <pre style={{ color: "#f85149", fontSize: "0.8rem", whiteSpace: "pre-wrap", padding: "0.5rem" }}>
          {error}
        </pre>
      </div>
    );
  }

  if (!result) {
    return <StatusBar>Press Cmd+Enter or click Run to execute</StatusBar>;
  }

  const { rows, columns, rowCount, bytesRead, pagesSkipped, durationMs } = result;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Stats bar */}
      <div style={statsBarStyle}>
        <span>{rowCount.toLocaleString()} rows</span>
        <span>{(bytesRead / 1024).toFixed(1)} KB read</span>
        <span>{pagesSkipped} pages skipped</span>
        <span>{durationMs.toFixed(1)} ms</span>
      </div>

      {/* Generated code */}
      {result.code && (
        <details style={{ padding: "0 0.75rem", fontSize: "0.8rem" }}>
          <summary style={{ color: "#8b949e", cursor: "pointer", padding: "0.25rem 0" }}>
            Generated descriptor
          </summary>
          <pre style={{ color: "#7ee787", padding: "0.25rem 0" }}>{result.code}</pre>
        </details>
      )}

      {/* Data table */}
      {rows.length > 0 ? (
        <div style={{ flex: 1, overflow: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.8rem" }}>
            <thead>
              <tr>
                <th style={thStyle}>#</th>
                {(columns || Object.keys(rows[0])).map((col) => (
                  <th key={col} style={thStyle}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? "transparent" : "#161b22" }}>
                  <td style={{ ...tdStyle, color: "#484f58" }}>{i + 1}</td>
                  {(columns || Object.keys(rows[0])).map((col) => (
                    <td key={col} style={tdStyle}>{formatCell(row[col])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={{ padding: "1rem", color: "#484f58", fontSize: "0.85rem" }}>
          Query returned 0 rows.
        </div>
      )}
    </div>
  );
}

function StatusBar({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ padding: "0.75rem", color: "#8b949e", fontSize: "0.85rem" }}>
      {children}
    </div>
  );
}

function formatCell(val: unknown): string {
  if (val === null || val === undefined) return "NULL";
  if (typeof val === "object") return JSON.stringify(val);
  return String(val);
}

const statsBarStyle: React.CSSProperties = {
  display: "flex",
  gap: "1.5rem",
  padding: "0.4rem 0.75rem",
  borderBottom: "1px solid #21262d",
  fontSize: "0.75rem",
  color: "#58a6ff",
};

const thStyle: React.CSSProperties = {
  position: "sticky",
  top: 0,
  background: "#161b22",
  borderBottom: "1px solid #21262d",
  padding: "0.35rem 0.6rem",
  textAlign: "left",
  color: "#8b949e",
  fontWeight: 600,
  whiteSpace: "nowrap",
};

const tdStyle: React.CSSProperties = {
  padding: "0.3rem 0.6rem",
  borderBottom: "1px solid #21262d10",
  whiteSpace: "nowrap",
  maxWidth: 300,
  overflow: "hidden",
  textOverflow: "ellipsis",
};
