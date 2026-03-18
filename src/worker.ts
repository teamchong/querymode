import type { Env, QueryDORpc, MasterDORpc } from "./types.js";
import { bigIntReplacer } from "./decode.js";
import { resolveBucket } from "./bucket.js";
import { QueryModeError } from "./errors.js";
import { MasterDO } from "./master-do.js";
import { QueryDO } from "./query-do.js";
import { FragmentDO } from "./fragment-do.js";

export { MasterDO, QueryDO, FragmentDO };

function json(body: unknown, status = 200, extraHeaders?: Record<string, string>): Response {
  return new Response(JSON.stringify(body, bigIntReplacer), {
    status, headers: { "content-type": "application/json", ...extraHeaders },
  });
}

/** Get the regional Query DO (typed as RPC). */
function getQueryDo(request: Request, env: Env): { rpc: QueryDORpc & { setRegion(region: string): Promise<void> }; regionName: string } {
  const cfRay = request.headers.get("cf-ray") ?? "";
  const datacenter = cfRay.split("-").pop() ?? "default";
  const regionName = `query-${datacenter}`;
  const queryId = env.QUERY_DO.idFromName(regionName);
  const rpc = env.QUERY_DO.get(queryId, { locationHint: datacenter as DurableObjectLocationHint }) as unknown as QueryDORpc & { setRegion(region: string): Promise<void> };
  return { rpc, regionName };
}

/** Get the Master DO (typed as RPC). Optionally shard by partition key for write parallelism. */
function getMasterDo(env: Env, partitionKey?: string): MasterDORpc {
  const name = partitionKey ? `master-${fnv1aHash(partitionKey)}` : "master";
  const masterId = env.MASTER_DO.idFromName(name);
  return env.MASTER_DO.get(masterId) as unknown as MasterDORpc;
}

/** FNV-1a hash → 4-character hex string for deterministic sharding. */
function fnv1aHash(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, "0").slice(0, 4);
}

/**
 * Cloudflare Worker entry point — the only HTTP layer.
 * Parses HTTP requests, calls DO RPC methods, returns HTTP responses.
 *
 * Routes:
 *   POST /query          → queryDo.queryRpc(body)
 *   POST /query/stream   → queryDo.streamRpc(body)
 *   POST /query/count    → queryDo.countRpc(body)
 *   POST /query/exists   → queryDo.existsRpc(body)
 *   POST /query/first    → queryDo.firstRpc(body)
 *   POST /query/explain  → queryDo.explainRpc(body)
 *   GET  /tables         → queryDo.listTablesRpc()
 *   GET  /meta?table=X   → queryDo.getMetaRpc(table)
 *   GET  /health?deep    → queryDo.diagnosticsRpc()
 *   POST /register-iceberg → queryDo.registerIcebergRpc(body)
 *   POST /write          → master.writeRpc(body)
 *   POST /refresh        → master.refreshRpc(body)
 *   POST /register       → master.registerRpc(id, region)
 */
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const requestId = request.headers.get("cf-ray") ?? crypto.randomUUID();
    const headers = { "x-querymode-request-id": requestId };

    try {
      // Health check
      if (url.pathname === "/health") {
        const base = {
          status: "ok",
          service: "querymode",
          region: request.headers.get("cf-ray")?.split("-").pop() ?? "unknown",
          timestamp: new Date().toISOString(),
        };

        if (url.searchParams.get("deep") === "true") {
          try {
            const { rpc, regionName } = getQueryDo(request, env);
            void rpc.setRegion(regionName).catch(() => {});
            const diagnostics = await rpc.diagnosticsRpc();
            return json({ ...base, diagnostics }, 200, headers);
          } catch (err) {
            return json({ ...base, diagnostics: { error: String(err) } }, 200, headers);
          }
        }

        return json(base, 200, headers);
      }

      // Direct R2 upload (local dev only — blocked in production)
      if (url.pathname === "/upload" && request.method === "POST") {
        if (!env.DEV_MODE) return json({ error: "upload disabled" }, 403);
        const key = url.searchParams.get("key");
        if (!key || key.includes("..") || key.startsWith("/")) {
          return json({ error: "invalid key" }, 400);
        }
        await resolveBucket(env, key).put(key, request.body);
        return json({ uploaded: key });
      }

      // ── Master DO routes ──────────────────────────────────────────────

      if (url.pathname === "/write") {
        const body = await request.json() as Record<string, unknown>;
        const partKey = typeof body.partitionKey === "string" ? body.partitionKey : undefined;
        const result = await getMasterDo(env, partKey).writeRpc(body);
        return json(result, 200, headers);
      }

      if (url.pathname === "/refresh") {
        const body = await request.json();
        const result = await getMasterDo(env).refreshRpc(body);
        return json(result, 200, headers);
      }

      if (url.pathname === "/register") {
        const { queryDoId, region } = (await request.json()) as { queryDoId: string; region: string };
        const result = await getMasterDo(env).registerRpc(queryDoId, region);
        return json(result, 200, headers);
      }

      // ── Query DO routes ───────────────────────────────────────────────

      if (url.pathname === "/query") {
        const { rpc, regionName } = getQueryDo(request, env);
        void rpc.setRegion(regionName).catch(() => {});
        const body = await request.json();
        const result = await rpc.queryRpc(body);
        result.requestId = requestId;
        return json(result, 200, headers);
      }

      if (url.pathname === "/query/stream") {
        const { rpc, regionName } = getQueryDo(request, env);
        void rpc.setRegion(regionName).catch(() => {});
        const body = await request.json();
        const stream = await rpc.streamRpc(body);
        return new Response(stream, {
          headers: { "content-type": "application/x-querymode-columnar", ...headers },
        });
      }

      if (url.pathname === "/query/count") {
        const { rpc, regionName } = getQueryDo(request, env);
        void rpc.setRegion(regionName).catch(() => {});
        const body = await request.json();
        const count = await rpc.countRpc(body);
        return json({ count }, 200, headers);
      }

      if (url.pathname === "/query/exists") {
        const { rpc, regionName } = getQueryDo(request, env);
        void rpc.setRegion(regionName).catch(() => {});
        const body = await request.json();
        const exists = await rpc.existsRpc(body);
        return json({ exists }, 200, headers);
      }

      if (url.pathname === "/query/first") {
        const { rpc, regionName } = getQueryDo(request, env);
        void rpc.setRegion(regionName).catch(() => {});
        const body = await request.json();
        const row = await rpc.firstRpc(body);
        return json({ row }, 200, headers);
      }

      if (url.pathname === "/query/explain") {
        const { rpc, regionName } = getQueryDo(request, env);
        void rpc.setRegion(regionName).catch(() => {});
        const body = await request.json();
        const result = await rpc.explainRpc(body);
        return json(result, 200, headers);
      }

      if (url.pathname === "/tables") {
        const { rpc, regionName } = getQueryDo(request, env);
        void rpc.setRegion(regionName).catch(() => {});
        const result = await rpc.listTablesRpc();
        return json(result, 200, headers);
      }

      if (url.pathname === "/meta") {
        const table = url.searchParams.get("table");
        if (!table) return json({ error: "Missing ?table= parameter" }, 400, headers);
        const { rpc, regionName } = getQueryDo(request, env);
        void rpc.setRegion(regionName).catch(() => {});
        const meta = await rpc.getMetaRpc(table);
        if (!meta) return json({ error: `Table "${table}" not found` }, 404, headers);
        return json(meta, 200, headers);
      }

      if (url.pathname === "/register-iceberg") {
        const { rpc, regionName } = getQueryDo(request, env);
        void rpc.setRegion(regionName).catch(() => {});
        const body = await request.json();
        const result = await rpc.registerIcebergRpc(body);
        return json(result, 200, headers);
      }

      return new Response("Not found. Routes: /query, /write, /refresh, /tables, /meta, /health", {
        status: 404,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (err instanceof SyntaxError) return json({ error: `Bad request: ${msg}` }, 400, headers);
      let status = 500;
      if (err instanceof QueryModeError) {
        if (err.code === "TABLE_NOT_FOUND" || err.code === "COLUMN_NOT_FOUND") status = 404;
        else if (err.code === "INVALID_FILTER" || err.code === "INVALID_FORMAT" || err.code === "INVALID_AGGREGATE" || err.code === "SCHEMA_MISMATCH") status = 400;
        else if (err.code === "QUERY_TIMEOUT" || err.code === "NETWORK_TIMEOUT") status = 504;
        else if (err.code === "MEMORY_EXCEEDED") status = 503;
      } else if (msg.includes("CAS failed")) {
        status = 409;
      } else if (msg.includes("not found")) {
        status = 404;
      }
      return json({ error: msg, ...(err instanceof QueryModeError && { code: err.code }) }, status, headers);
    }
  },
};
