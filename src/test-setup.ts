/**
 * Polyfill Cloudflare Workers globals for vitest.
 * DurableObject is a base class provided by the Workers runtime —
 * tests run in Node and need a minimal stand-in.
 */
if (typeof globalThis.DurableObject === "undefined") {
  (globalThis as Record<string, unknown>).DurableObject = class DurableObject {
    protected ctx: unknown;
    protected env: unknown;
    constructor(ctx: unknown, env: unknown) {
      this.ctx = ctx;
      this.env = env;
    }
  };
}
