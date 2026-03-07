const d = new Diagram({ direction: "TB" });

// ── Row 0: Client + SDK ──
const client = d.addEllipse("Client\nPOST /query", { row: 0, col: 1, color: "users" });
const sdk = d.addBox("SDK: QueryMode.remote(DO) | QueryMode.local()", { row: 0, col: 2, color: "users" });

// ── Row 1: Router ──
const worker = d.addBox("Cloudflare Worker (Router)\nExtracts CF-Ray datacenter, sets x-querymode-region\n/query | /query/stream | /write | /refresh", { row: 1, col: 1, color: "backend" });

// ── Row 2: Master DO + Regional Query DOs (same row = compact) ──
const masterDO = d.addBox("Master DO\n(Single Writer)\nSQLite: TableMeta\nReads footers from R2\nBroadcasts invalidations\nReturns tableVersions", { row: 2, col: 0, color: "orchestration" });
const sjc = d.addBox("Query DO — SJC\nFooter cache (~4KB/table)\nWASM Engine (Zig SIMD)\nStreaming operator pipeline\nVIP eviction (zell)\nCoalesced Range reads\nrefreshStaleTables()\nColumnar binary streaming", { row: 2, col: 1, color: "frontend" });
const nrt = d.addBox("Query DO — NRT\nFooter cache (~4KB/table)\nWASM Engine (Zig SIMD)\nStreaming operator pipeline\nVIP eviction (zell)\nCoalesced Range reads\nrefreshStaleTables()\nColumnar binary streaming", { row: 2, col: 2, color: "frontend" });
const ams = d.addBox("Query DO — AMS\nFooter cache (~4KB/table)\nWASM Engine (Zig SIMD)\nStreaming operator pipeline\nVIP eviction (zell)\nCoalesced Range reads\nrefreshStaleTables()\nColumnar binary streaming", { row: 2, col: 3, color: "frontend" });

// ── Row 3: R2 Data Lake ──
const r2 = d.addBox("R2 Data Lake\nLance files + _versions/*.manifest\nCoalesced Range reads (~10ms via R2 bindings)\nFree egress, $0.36/M Class B ops", { row: 3, col: 1, color: "storage" });

// ── Row 4: Pipeline (numbered steps) ──
const pipeFetch = d.addBox("1. R2 Page Fetch\n(coalesced Range)", { row: 4, col: 0, color: "storage" });
const pipeWasm = d.addBox("2. PageProcessor\nWASM (Zig SIMD)", { row: 4, col: 1, color: "ai" });
const pipeAccum = d.addBox("3. Accumulate Partial\nResults (partial-agg.ts)", { row: 4, col: 2, color: "database" });
const pipeReturn = d.addBox("4. Return RPC result\n(zero serialization)", { row: 4, col: 3, color: "frontend" });

// ── Row 5-7: Fragment DO Pool ──
const coord = d.addBox("Query DO\n(Coordinator)", { row: 5, col: 0, color: "frontend" });
const frag0 = d.addBox("Fragment DO\nslot-0: frags 0..99\n+ footer cache", { row: 5, col: 1, color: "external" });
const frag1 = d.addBox("Fragment DO\nslot-1: frags 100..199\n+ footer cache", { row: 6, col: 1, color: "external" });
const frag2 = d.addBox("Fragment DO\nslot-2: frags 200..299\n+ footer cache", { row: 7, col: 1, color: "external" });
const merge = d.addBox("mergeQueryResults()\nk-way merge | partial-agg merge", { row: 6, col: 2, color: "database" });

// ── Groups ──
d.addGroup("Regional Query DOs  (1 per datacenter)", [sjc, nrt, ams], { strokeColor: "#868e96" });
d.addGroup("Page-Streaming Pipeline (inside Query DO + Fragment DO)  →", [pipeFetch, pipeWasm, pipeAccum, pipeReturn], { strokeColor: "#9c36b5" });
d.addGroup("TB+ Scale: Fragment DO Pool (frag-{region}-slot-{N}, max 20)", [coord, frag0, frag1, frag2, merge], { strokeColor: "#e03131" });

// ── Connections: Client → Worker ──
d.connect(client, worker, "HTTPS request");
d.connect(sdk, worker, "QueryMode.remote()");

// ── Connections: Worker → services ──
d.connect(worker, masterDO, "/write, /refresh", { strokeColor: "#c92a2a" });
d.connect(worker, sjc, "", { strokeColor: "#1971c2" });
d.connect(worker, nrt, "", { strokeColor: "#1971c2" });
d.connect(worker, ams, "", { strokeColor: "#1971c2" });

// ── Connections: Master DO broadcast (dashed) ──
d.connect(masterDO, sjc, "broadcast", { style: "dashed", strokeColor: "#e03131" });
d.connect(masterDO, nrt, "", { style: "dashed", strokeColor: "#e03131" });
d.connect(masterDO, ams, "", { style: "dashed", strokeColor: "#e03131" });

// ── Connections: All Query DOs register on wake (dotted) ──
d.connect(sjc, masterDO, "register on wake\n+ refreshStaleTables()", { style: "dotted", strokeColor: "#2f9e44" });
d.connect(nrt, masterDO, "", { style: "dotted", strokeColor: "#2f9e44" });
d.connect(ams, masterDO, "", { style: "dotted", strokeColor: "#2f9e44" });

// ── Connections: Query DOs → R2 ──
d.connect(sjc, r2, "Coalesced Range reads\n(merge within 64KB gap)", { strokeColor: "#f08c00" });
d.connect(nrt, r2, "", { strokeColor: "#f08c00" });
d.connect(ams, r2, "", { strokeColor: "#f08c00" });
d.connect(masterDO, r2, "read footer (40B)", { strokeColor: "#c92a2a" });

// ── Connections: R2 → Pipeline + Fragment Pool (vertical connectors) ──
d.connect(r2, pipeFetch, "page reads", { strokeColor: "#f08c00" });
d.connect(r2, coord, "TB+ scale", { strokeColor: "#1971c2" });

// ── Connections: Fragment DO Pool ──
d.connect(coord, frag0, "fan-out", { strokeColor: "#e03131" });
d.connect(coord, frag1, "", { strokeColor: "#e03131" });
d.connect(coord, frag2, "", { strokeColor: "#e03131" });
d.connect(frag0, merge, "", { strokeColor: "#2f9e44" });
d.connect(frag1, merge, "", { strokeColor: "#2f9e44" });
d.connect(frag2, merge, "", { strokeColor: "#2f9e44" });

// ── Legend (top-right, clear of diagram content) ──
d.addText("Legend\n\n--- Solid arrow: data flow\n- - Dashed: broadcast invalidation\n... Dotted: register + refreshStale\n\nPurple = Worker Router\nBlue = Query DOs + SDK\nRed/Coral = Master DO / Fragment DOs\nYellow = R2 Data Lake\nGreen = Results / Merge\nMagenta = WASM / PageProcessor", { x: 1260, y: 60, fontSize: 11 });

// ── Bounded Prefetch (below Pipeline group, left side) ──
d.addText("Bounded Prefetch:\nfetch page N+1 while\nWASM processes page N", { x: 300, y: 1210, fontSize: 11, color: "cache" });

return d.render({ format: ["excalidraw", "png"], path: "docs/architecture/querymode-architecture" });
