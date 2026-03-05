/**
 * Polyfill for `cloudflare:workers` — used in vitest only.
 * The real module is provided by the Workers runtime at deploy time.
 */
export class DurableObject {
  protected ctx: unknown;
  protected env: unknown;
  constructor(ctx: unknown, env: unknown) {
    this.ctx = ctx;
    this.env = env;
  }
}
