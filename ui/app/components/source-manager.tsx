"use client";

import { useState, useEffect, useCallback, useRef } from "react";

interface R2Object {
  key: string;
  size: number;
  lastModified: string;
}

export function SourceManager() {
  const [objects, setObjects] = useState<R2Object[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/sources");
      if (res.ok) setObjects(await res.json());
    } catch {} finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const upload = useCallback(async (file: File) => {
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch("/api/sources", { method: "POST", body: formData });
      if (res.ok) await refresh();
    } catch {} finally {
      setUploading(false);
    }
  }, [refresh]);

  const deleteObject = useCallback(async (key: string) => {
    if (!confirm(`Delete "${key}"?`)) return;
    await fetch(`/api/sources?key=${encodeURIComponent(key)}`, { method: "DELETE" });
    await refresh();
  }, [refresh]);

  return (
    <div style={{ padding: "1rem", maxWidth: 800, margin: "0 auto" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
        <h2 style={{ fontSize: "1rem", fontWeight: 600 }}>R2 Data Sources</h2>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button onClick={refresh} style={actionBtnStyle}>↻ Refresh</button>
          <button onClick={() => fileRef.current?.click()} disabled={uploading} style={uploadBtnStyle}>
            {uploading ? "Uploading..." : "↑ Upload"}
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".csv,.json,.parquet,.lance"
            style={{ display: "none" }}
            onChange={(e) => { if (e.target.files?.[0]) upload(e.target.files[0]); }}
          />
        </div>
      </div>

      {loading ? (
        <div style={{ color: "#8b949e" }}>Loading...</div>
      ) : objects.length === 0 ? (
        <div style={{ color: "#484f58", padding: "2rem", textAlign: "center" }}>
          No datasets found. Upload CSV, JSON, Parquet, or Lance files.
        </div>
      ) : (
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
          <thead>
            <tr>
              <th style={thStyle}>Key</th>
              <th style={thStyle}>Size</th>
              <th style={thStyle}>Modified</th>
              <th style={thStyle}></th>
            </tr>
          </thead>
          <tbody>
            {objects.map((obj) => (
              <tr key={obj.key}>
                <td style={tdStyle}>{obj.key}</td>
                <td style={tdStyle}>{formatSize(obj.size)}</td>
                <td style={tdStyle}>{new Date(obj.lastModified).toLocaleString()}</td>
                <td style={tdStyle}>
                  <button onClick={() => deleteObject(obj.key)} style={deleteBtnStyle}>
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function formatSize(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
  return `${bytes} B`;
}

const thStyle: React.CSSProperties = {
  borderBottom: "1px solid #21262d",
  padding: "0.5rem 0.75rem",
  textAlign: "left",
  color: "#8b949e",
  fontWeight: 600,
};

const tdStyle: React.CSSProperties = {
  padding: "0.5rem 0.75rem",
  borderBottom: "1px solid #21262d",
};

const actionBtnStyle: React.CSSProperties = {
  background: "#21262d",
  color: "#c9d1d9",
  border: "1px solid #30363d",
  borderRadius: 6,
  padding: "0.35rem 0.75rem",
  cursor: "pointer",
  fontSize: "0.8rem",
};

const uploadBtnStyle: React.CSSProperties = {
  background: "#238636",
  color: "white",
  border: "none",
  borderRadius: 6,
  padding: "0.35rem 0.75rem",
  cursor: "pointer",
  fontSize: "0.8rem",
  fontWeight: 600,
};

const deleteBtnStyle: React.CSSProperties = {
  background: "none",
  color: "#f85149",
  border: "none",
  cursor: "pointer",
  fontSize: "0.8rem",
};
