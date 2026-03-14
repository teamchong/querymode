"use client";

import { useEffect, useState } from "react";

export interface ColumnInfo {
  name: string;
  dtype: string;
  nullCount: number;
  totalRows: number;
}

export interface TableSchema {
  name: string;
  columns: ColumnInfo[];
  totalRows: number;
  fileSize: number;
  fragmentCount: number;
}

interface Props {
  schemas: TableSchema[];
  onRefresh: () => void;
  onInsert: (snippet: string) => void;
}

export function SchemaPanel({ schemas, onRefresh, onInsert }: Props) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  useEffect(() => { onRefresh(); }, [onRefresh]);

  const toggle = (name: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  return (
    <div style={panelStyle}>
      <div style={panelHeaderStyle}>
        <span>Schema</span>
        <button onClick={onRefresh} style={refreshBtnStyle} title="Refresh schema">↻</button>
      </div>

      {schemas.length === 0 && (
        <div style={{ padding: "1rem", color: "#484f58", fontSize: "0.8rem" }}>
          No tables found. Upload data in Sources tab.
        </div>
      )}

      {schemas.map((table) => (
        <div key={table.name}>
          <div
            onClick={() => toggle(table.name)}
            style={tableRowStyle}
          >
            <span style={{ color: "#8b949e" }}>{expanded.has(table.name) ? "▼" : "▶"}</span>
            <span style={{ fontWeight: 600 }}>{table.name}</span>
            <span style={{ color: "#484f58", marginLeft: "auto", fontSize: "0.75rem" }}>
              {formatRows(table.totalRows)}
            </span>
          </div>

          {expanded.has(table.name) && (
            <div style={{ paddingLeft: "1.25rem" }}>
              {table.columns.map((col) => (
                <div
                  key={col.name}
                  style={colRowStyle}
                  onClick={() => onInsert(`qm.table("${table.name}").select("${col.name}")`)}
                  title="Click to insert"
                >
                  <span>{col.name}</span>
                  <span style={{ color: dtypeColor(col.dtype), fontSize: "0.75rem" }}>
                    {col.dtype}
                  </span>
                </div>
              ))}
              <div
                style={{ ...colRowStyle, color: "#58a6ff", cursor: "pointer" }}
                onClick={() => onInsert(`qm.table("${table.name}").limit(100).execute()`)}
              >
                ↳ preview 100 rows
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function formatRows(n: number): string {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

function dtypeColor(dtype: string): string {
  if (dtype.startsWith("int") || dtype.startsWith("uint") || dtype === "float32" || dtype === "float64") return "#d2a8ff";
  if (dtype === "utf8" || dtype === "string") return "#7ee787";
  if (dtype === "bool") return "#ffa657";
  return "#8b949e";
}

const panelStyle: React.CSSProperties = {
  width: 240,
  minWidth: 240,
  borderRight: "1px solid #21262d",
  overflow: "auto",
  background: "#0d1117",
};

const panelHeaderStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  padding: "0.5rem 0.75rem",
  borderBottom: "1px solid #21262d",
  fontSize: "0.8rem",
  color: "#8b949e",
  textTransform: "uppercase",
  letterSpacing: "0.05em",
};

const refreshBtnStyle: React.CSSProperties = {
  background: "none",
  border: "none",
  color: "#8b949e",
  cursor: "pointer",
  fontSize: "1rem",
};

const tableRowStyle: React.CSSProperties = {
  display: "flex",
  gap: "0.5rem",
  alignItems: "center",
  padding: "0.35rem 0.75rem",
  cursor: "pointer",
  fontSize: "0.85rem",
};

const colRowStyle: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  padding: "0.2rem 0.75rem",
  fontSize: "0.8rem",
  cursor: "pointer",
  color: "#c9d1d9",
};
