export async function GET() {
  try {
    const env = (process as unknown as { env: { QUERYMODE: Fetcher } }).env;
    const res = await env.QUERYMODE.fetch("https://querymode/api/schema");

    if (!res.ok) {
      return new Response(await res.text(), { status: res.status });
    }

    return new Response(res.body, {
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return Response.json([]);
  }
}
