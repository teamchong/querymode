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
 */
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // Health check
    if (url.pathname === "/health") {
      return new Response(JSON.stringify({ status: "ok", service: "edgeq" }), {
        headers: { "content-type": "application/json" },
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

      // Pass region name via header so the Query DO can store it for Master registration
      const reqWithRegion = new Request(request.url, request);
      reqWithRegion.headers.set("x-edgeq-region", regionName);
      return queryDo.fetch(reqWithRegion);
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
