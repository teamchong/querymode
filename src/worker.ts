import type { Env, QueryDORpc, MasterDORpc } from "./types.js";
import { bigIntReplacer } from "./decode.js";
import { resolveBucket } from "./bucket.js";
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

/** Get the Master DO (typed as RPC). */
function getMasterDo(env: Env): MasterDORpc {
  const masterId = env.MASTER_DO.idFromName("master");
  return env.MASTER_DO.get(masterId) as unknown as MasterDORpc;
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
            rpc.setRegion(regionName).catch(() => {});
            const diagnostics = await rpc.diagnosticsRpc();
            return json({ ...base, diagnostics }, 200, headers);
          } catch (err) {
            return json({ ...base, diagnostics: { error: String(err) } }, 200, headers);
          }
        }

        return json(base, 200, headers);
      }

      // Direct R2 upload (local dev only)
      if (url.pathname === "/upload" && request.method === "POST") {
        const key = url.searchParams.get("key");
        if (!key) return new Response("Missing ?key=", { status: 400 });
        await resolveBucket(env, key).put(key, request.body);
        return json({ uploaded: key });
      }

      // ── Master DO routes ──────────────────────────────────────────────

      if (url.pathname === "/write") {
        const body = await request.json();
        const result = await getMasterDo(env).writeRpc(body);
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
        rpc.setRegion(regionName).catch(() => {});
        const body = await request.json();
        const result = await rpc.queryRpc(body);
        result.requestId = requestId;
        return json(result, 200, headers);
      }

      if (url.pathname === "/query/stream") {
        const { rpc, regionName } = getQueryDo(request, env);
        rpc.setRegion(regionName).catch(() => {});
        const body = await request.json();
        const stream = await rpc.streamRpc(body);
        return new Response(stream, {
          headers: { "content-type": "application/x-querymode-columnar", ...headers },
        });
      }

      if (url.pathname === "/query/count") {
        const { rpc, regionName } = getQueryDo(request, env);
        rpc.setRegion(regionName).catch(() => {});
        const body = await request.json();
        const count = await rpc.countRpc(body);
        return json({ count }, 200, headers);
      }

      if (url.pathname === "/query/exists") {
        const { rpc, regionName } = getQueryDo(request, env);
        rpc.setRegion(regionName).catch(() => {});
        const body = await request.json();
        const exists = await rpc.existsRpc(body);
        return json({ exists }, 200, headers);
      }

      if (url.pathname === "/query/first") {
        const { rpc, regionName } = getQueryDo(request, env);
        rpc.setRegion(regionName).catch(() => {});
        const body = await request.json();
        const row = await rpc.firstRpc(body);
        return json({ row }, 200, headers);
      }

      if (url.pathname === "/query/explain") {
        const { rpc, regionName } = getQueryDo(request, env);
        rpc.setRegion(regionName).catch(() => {});
        const body = await request.json();
        const result = await rpc.explainRpc(body);
        return json(result, 200, headers);
      }

      if (url.pathname === "/tables") {
        const { rpc, regionName } = getQueryDo(request, env);
        rpc.setRegion(regionName).catch(() => {});
        const result = await rpc.listTablesRpc();
        return json(result, 200, headers);
      }

      if (url.pathname === "/meta") {
        const table = url.searchParams.get("table");
        if (!table) return json({ error: "Missing ?table= parameter" }, 400, headers);
        const { rpc, regionName } = getQueryDo(request, env);
        rpc.setRegion(regionName).catch(() => {});
        const meta = await rpc.getMetaRpc(table);
        if (!meta) return json({ error: `Table "${table}" not found` }, 404, headers);
        return json(meta, 200, headers);
      }

      if (url.pathname === "/register-iceberg") {
        const { rpc, regionName } = getQueryDo(request, env);
        rpc.setRegion(regionName).catch(() => {});
        const body = await request.json();
        const result = await rpc.registerIcebergRpc(body);
        return json(result, 200, headers);
      }

      return new Response("Not found. Routes: /query, /write, /refresh, /tables, /meta, /health", {
        status: 404,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      const status = msg.includes("CAS failed") ? 409 : msg.includes("not found") ? 404 : 500;
      return json({ error: msg }, status, headers);
    }
  },
};
