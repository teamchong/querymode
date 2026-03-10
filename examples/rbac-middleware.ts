/**
 * RBAC middleware example — row-level and column-level security via filter injection.
 *
 * Security isn't a feature to build. It's a filter to inject.
 *
 * This Worker middleware authenticates the request, then modifies the query
 * descriptor before it reaches QueryDO. Each tenant only sees their own data.
 * Sensitive columns are stripped for non-admin roles.
 *
 * Works with:
 *   - Cloudflare Access (JWT in cf-access-jwt-assertion header)
 *   - Any JWT issuer (Auth0, Clerk, Supabase Auth, etc.)
 *   - API keys (look up role from D1/KV)
 */

// ─── Types ───────────────────────────────────────────────────────────────

interface User {
  id: string;
  tenantId: string;
  role: "admin" | "analyst" | "viewer";
}

interface Env {
  QUERY_DO: DurableObjectNamespace;
  // Optional: store API keys and roles
  // AUTH_KV: KVNamespace;
}

// ─── Column policies ────────────────────────────────────────────────────

/** Columns hidden from non-admin roles */
const RESTRICTED_COLUMNS: Record<string, string[]> = {
  viewer: ["revenue", "cost", "margin", "salary", "ssn", "email"],
  analyst: ["ssn", "salary"],
  admin: [], // admins see everything
};

// ─── Auth ────────────────────────────────────────────────────────────────

/** Extract user from JWT. Replace with your auth provider's verification. */
async function authenticate(request: Request): Promise<User> {
  // Cloudflare Access: JWT is in cf-access-jwt-assertion header
  const jwt = request.headers.get("cf-access-jwt-assertion")
    ?? request.headers.get("authorization")?.replace("Bearer ", "");

  if (!jwt) throw new Error("No auth token");

  // In production: verify JWT signature against your JWKS endpoint
  // const payload = await verifyJwt(jwt, env.JWKS_URL);
  // For this example, decode without verification:
  const payload = JSON.parse(atob(jwt.split(".")[1]));

  return {
    id: payload.sub,
    tenantId: payload.tenant_id ?? payload.org_id ?? "default",
    role: payload.role ?? "viewer",
  };
}

// ─── Middleware ──────────────────────────────────────────────────────────

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // 1. Authenticate
    let user: User;
    try {
      user = await authenticate(request);
    } catch {
      return new Response(JSON.stringify({ error: "Unauthorized" }), {
        status: 401,
        headers: { "Content-Type": "application/json" },
      });
    }

    // 2. Parse the query descriptor from the request body
    const body = await request.json() as { descriptor: Record<string, unknown> };
    const descriptor = body.descriptor;

    // 3. Row-level security: inject tenant filter
    //    Every query automatically scoped to the user's tenant.
    if (user.role !== "admin") {
      const filters = (descriptor.filters ?? []) as { column: string; op: string; value: unknown }[];
      filters.push({ column: "tenant_id", op: "eq", value: user.tenantId });
      descriptor.filters = filters;
    }

    // 4. Column-level security: strip restricted columns from projections
    const restricted = new Set(RESTRICTED_COLUMNS[user.role] ?? []);
    if (restricted.size > 0) {
      const projections = descriptor.projections as string[] | undefined;
      if (projections && projections.length > 0) {
        descriptor.projections = projections.filter((c: string) => !restricted.has(c));
      }
    }

    // 5. Forward to QueryDO — the query runs with security filters baked in
    const doId = env.QUERY_DO.idFromName("default");
    const queryDo = env.QUERY_DO.get(doId);
    const result = await (queryDo as unknown as { queryRpc(d: unknown): Promise<unknown> })
      .queryRpc(descriptor);

    return new Response(JSON.stringify(result), {
      headers: {
        "Content-Type": "application/json",
        "X-QueryMode-User": user.id,
        "X-QueryMode-Tenant": user.tenantId,
      },
    });
  },
};

// ─── What this gives you ─────────────────────────────────────────────────
//
// 1. Row-level security  = injected filter (tenant_id = user's tenant)
// 2. Column-level security = stripped projections (no salary/ssn for viewers)
// 3. Audit trail = add env.AUDIT_LOG.writeDataPoint() before the query
// 4. Rate limiting = add env.RATE_LIMITER.check() before the query
//
// No permission tables. No GRANT/REVOKE. No policy engine.
// Security is just filters and projections — things QueryMode already does.
