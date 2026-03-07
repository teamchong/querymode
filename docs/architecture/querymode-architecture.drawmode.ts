
const d = new Diagram({ theme: "sketch" });

const client = d.addBox("Client SDK", { row: 0, col: 1, color: "frontend" });
const sql = d.addBox("SQL Frontend", { row: 0, col: 3, color: "frontend" });
const worker = d.addBox("CF Worker", { row: 1, col: 2, color: "backend" });
const master = d.addBox("MasterDO", { row: 2, col: 0, color: "orchestration" });
const queryDO = d.addBox("QueryDO", { row: 2, col: 2, color: "orchestration" });
const fragDO = d.addBox("FragmentDO", { row: 2, col: 4, color: "orchestration" });
const wasm = d.addBox("Zig WASM Engine", { row: 3, col: 2, color: "ai" });
const scan = d.addBox("Scan", { row: 4, col: 0, color: "cache" });
const filter = d.addBox("Filter", { row: 4, col: 1, color: "cache" });
const agg = d.addBox("Agg", { row: 4, col: 2, color: "cache" });
const sort = d.addBox("Sort", { row: 4, col: 3, color: "cache" });
const proj = d.addBox("Project", { row: 4, col: 4, color: "cache" });
const r2 = d.addBox("R2 Storage", { row: 5, col: 1, color: "storage" });
const fmt = d.addBox("Parquet/Lance/Iceberg", { row: 5, col: 3, color: "storage" });

d.connect(client, worker, "query");
d.connect(sql, worker, "SQL");
d.connect(worker, master, "register");
d.connect(worker, queryDO, "query");
d.connect(queryDO, fragDO, "fan-out");
d.connect(queryDO, wasm, "decode");
d.connect(wasm, scan, "pull");
d.connect(scan, filter);
d.connect(filter, agg);
d.connect(agg, sort);
d.connect(sort, proj);
d.connect(scan, r2, "read");
d.connect(r2, fmt);

return d.render({ format: ["excalidraw", "svg", "png"], path: "/Users/steven_chong/Downloads/repos/querymode/docs/architecture/querymode-architecture" });
