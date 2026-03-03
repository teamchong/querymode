import type { Env } from "./types.js";
import { MasterDO } from "./master-do.js";
import { QueryDO } from "./query-do.js";
import { FragmentDO } from "./fragment-do.js";

export { MasterDO, QueryDO, FragmentDO };

/**
 * Cloudflare Worker entry point.
 * Routes incoming requests to the appropriate Durable Object.
 *
 * Routes:
 *   POST /query          → Regional Query DO (nearest to caller)
 *   POST /write          → Master DO (single writer)
 *   POST /refresh        → Master DO (re-read footer from R2)
 *   GET  /tables         → Regional Query DO (list cached tables)
 *   GET  /meta?table=X   → Regional Query DO (table metadata)
 *   GET  /health         → Health check (optional ?deep=true for diagnostics)
 */
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const requestId = request.headers.get("cf-ray") ?? crypto.randomUUID();

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
          const cfRay = request.headers.get("cf-ray") ?? "";
          const datacenter = cfRay.split("-").pop() ?? "default";
          const regionName = `query-${datacenter}`;
          const queryId = env.QUERY_DO.idFromName(regionName);
          const queryDo = env.QUERY_DO.get(queryId, { locationHint: datacenter as DurableObjectLocationHint });

          const diagResp = await queryDo.fetch(new Request("http://internal/diagnostics"));
          const diagnostics = await diagResp.json();
          return new Response(JSON.stringify({ ...base, diagnostics }), {
            headers: { "content-type": "application/json", "x-querymode-request-id": requestId },
          });
        } catch (err) {
          return new Response(JSON.stringify({ ...base, diagnostics: { error: String(err) } }), {
            headers: { "content-type": "application/json", "x-querymode-request-id": requestId },
          });
        }
      }

      return new Response(JSON.stringify(base), {
        headers: { "content-type": "application/json", "x-querymode-request-id": requestId },
      });
    }

    // Write operations go to the Master DO (single writer)
    if (url.pathname === "/write" || url.pathname === "/refresh") {
      const masterId = env.MASTER_DO.idFromName("master");
      const master = env.MASTER_DO.get(masterId);
      return master.fetch(request);
    }

    // Read operations go to the nearest regional Query DO
    if (
      url.pathname === "/query" ||
      url.pathname === "/query/stream" ||
      url.pathname === "/tables" ||
      url.pathname === "/meta"
    ) {
      // Use a deterministic name per region so each region gets one Query DO.
      // The CF-Ray header contains the datacenter code (e.g., "SJC", "NRT").
      const cfRay = request.headers.get("cf-ray") ?? "";
      const datacenter = cfRay.split("-").pop() ?? "default";
      const regionName = `query-${datacenter}`;

      const queryId = env.QUERY_DO.idFromName(regionName);
      const queryDo = env.QUERY_DO.get(queryId, { locationHint: datacenter as DurableObjectLocationHint });

      // Pass region name + request ID via headers
      const reqWithRegion = new Request(request.url, request);
      reqWithRegion.headers.set("x-querymode-region", regionName);
      reqWithRegion.headers.set("x-querymode-request-id", requestId);

      const resp = await queryDo.fetch(reqWithRegion);
      const respWithId = new Response(resp.body, resp);
      respWithId.headers.set("x-querymode-request-id", requestId);
      return respWithId;
    }

    // Register a regional Query DO with the Master DO
    if (url.pathname === "/register") {
      const masterId = env.MASTER_DO.idFromName("master");
      const master = env.MASTER_DO.get(masterId);
      return master.fetch(request);
    }

    return new Response("Not found. Routes: /query, /write, /refresh, /tables, /meta, /health", {
      status: 404,
    });
  },
};
